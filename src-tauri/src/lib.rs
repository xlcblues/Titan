use rusqlite::{params, Connection};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::{Emitter, Manager, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AppInfo {
    name: &'static str,
    version: &'static str,
    engine: &'static str,
    gpu_budget: &'static str,
    target_platform: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Project {
    id: String,
    name: String,
    status: String,
    settings_json: String,
    image_count: i64,
    created_at: i64,
    updated_at: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageRecord {
    id: String,
    project_id: String,
    path: String,
    filename: Option<String>,
    width: Option<i64>,
    height: Option<i64>,
    size: Option<i64>,
    status: String,
    thumbnail_path: Option<String>,
    sort_order: i64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TaskInfo {
    id: String,
    project_id: String,
    kind: String,
    status: String,
    stage: String,
    progress: f64,
    started_at: Option<i64>,
    finished_at: Option<i64>,
    error: Option<String>,
    latest: Option<Value>,
}

struct TaskRegistry(Mutex<HashMap<String, Child>>);

fn kill_child(child: &mut Child) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = child.kill();
    }
}

struct Db(Mutex<Connection>);

impl Db {
    fn open(path: &Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let conn = Connection::open(path).map_err(|error| error.to_string())?;
        conn.pragma_update(None, "foreign_keys", true)
            .map_err(|error| error.to_string())?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS projects (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'idle',
              settings_json TEXT NOT NULL DEFAULT '{}',
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              path TEXT NOT NULL,
              filename TEXT,
              width INTEGER,
              height INTEGER,
              size INTEGER,
              status TEXT NOT NULL DEFAULT 'pending',
              thumbnail_path TEXT,
              sort_order INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS tasks (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              type TEXT NOT NULL,
              status TEXT NOT NULL,
              stage TEXT NOT NULL DEFAULT 'data',
              progress REAL NOT NULL DEFAULT 0,
              started_at INTEGER,
              finished_at INTEGER,
              error TEXT,
              latest_json TEXT,
              config_json TEXT
            );
            CREATE TABLE IF NOT EXISTS metrics (
              id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
              step INTEGER NOT NULL,
              psnr REAL,
              ssim REAL,
              lpips REAL,
              loss REAL,
              vram_mb INTEGER,
              created_at INTEGER NOT NULL
            );
            ",
        )
        .map_err(|error| error.to_string())?;
        ensure_task_columns(&conn)?;
        Ok(Self(Mutex::new(conn)))
    }
}

fn ensure_task_columns(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(tasks)")
        .map_err(|error| error.to_string())?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| error.to_string())?;
    let existing: Vec<String> = rows
        .collect::<Result<_, _>>()
        .map_err(|error| error.to_string())?;

    const MIGRATIONS: &[(&str, &str)] = &[
        ("stage", "TEXT NOT NULL DEFAULT 'data'"),
        ("progress", "REAL NOT NULL DEFAULT 0"),
        ("started_at", "INTEGER"),
        ("finished_at", "INTEGER"),
        ("error", "TEXT"),
        ("latest_json", "TEXT"),
        ("config_json", "TEXT"),
    ];

    for (column, definition) in MIGRATIONS {
        if existing
            .iter()
            .any(|item| item.eq_ignore_ascii_case(column))
        {
            continue;
        }
        conn.execute(
            &format!("ALTER TABLE tasks ADD COLUMN {column} {definition}"),
            [],
        )
        .map_err(|error| error.to_string())?;
    }

    Ok(())
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn app_data_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path().app_data_dir().map_err(|error| error.to_string())
}

fn project_dir(app: &tauri::AppHandle, project_id: &str) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join("projects").join(project_id))
}

fn task_project_id(app: &tauri::AppHandle, task_id: &str) -> Result<String, String> {
    let db = app.state::<Db>();
    let conn = db.0.lock().unwrap();
    conn.query_row(
        "SELECT project_id FROM tasks WHERE id = ?1",
        [task_id],
        |row| row.get(0),
    )
    .map_err(|error| error.to_string())
}

fn resolve_python() -> Result<String, String> {
    if let Ok(custom) = std::env::var("TITAN_PYTHON") {
        if Path::new(&custom).exists() {
            return Ok(custom);
        }
    }
    let candidates = ["python", "python3"];
    for candidate in candidates {
        let mut command = Command::new(candidate);
        command.arg("-c").arg("import sys; print(sys.version_info[0], end='')");
        if command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .and_then(|mut child| child.wait())
            .map(|status| status.success())
            .unwrap_or(false)
        {
            return Ok(candidate.to_string());
        }
    }
    Err("未找到 Python 环境，请安装 Python 3.10+ 并配置 TITAN_PYTHON 环境变量".into())
}

fn write_control(project_dir: &Path, command: &str) -> Result<(), String> {
    fs::create_dir_all(project_dir).map_err(|error| error.to_string())?;
    let payload = json!({ "command": command, "at": now_millis() });
    let raw = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    fs::write(project_dir.join("control.json"), raw).map_err(|error| error.to_string())
}

#[tauri::command]
fn app_info() -> AppInfo {
    AppInfo {
        name: "Titan",
        version: env!("CARGO_PKG_VERSION"),
        engine: "Facet Radiance Field",
        gpu_budget: "8GB",
        target_platform: "Windows",
    }
}

#[tauri::command]
fn create_project(
    db: State<'_, Db>,
    name: String,
    settings: Option<serde_json::Value>,
) -> Result<Project, String> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err("项目名称不能为空".into());
    }

    let id = uuid::Uuid::new_v4().to_string();
    let now = now_millis();
    let settings_json = settings
        .map(|value| value.to_string())
        .unwrap_or_else(|| "{}".into());
    let conn = db.0.lock().unwrap();
    conn.execute(
        "INSERT INTO projects (id, name, status, settings_json, created_at, updated_at)
         VALUES (?1, ?2, 'idle', ?3, ?4, ?4)",
        params![id, name, settings_json, now],
    )
    .map_err(|error| error.to_string())?;

    Ok(Project {
        id,
        name,
        status: "idle".into(),
        settings_json,
        image_count: 0,
        created_at: now,
        updated_at: now,
    })
}

#[tauri::command]
fn list_projects(db: State<'_, Db>) -> Result<Vec<Project>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT
               p.id,
               p.name,
               p.status,
               p.settings_json,
               p.created_at,
               p.updated_at,
               (SELECT COUNT(*) FROM images i WHERE i.project_id = p.id)
             FROM projects p
             ORDER BY p.updated_at DESC",
        )
        .map_err(|error| error.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(Project {
                id: row.get(0)?,
                name: row.get(1)?,
                status: row.get(2)?,
                settings_json: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
                image_count: row.get(6)?,
            })
        })
        .map_err(|error| error.to_string())?;

    let mut projects = Vec::new();
    for row in rows {
        projects.push(row.map_err(|error| error.to_string())?);
    }
    Ok(projects)
}

#[tauri::command]
fn delete_project(app: tauri::AppHandle, project_id: String) -> Result<(), String> {
    let task_ids: Vec<String> = {
        let db = app.state::<Db>();
        let conn = db.0.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id FROM tasks WHERE project_id = ?1")
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map([&project_id], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row.map_err(|error| error.to_string())?);
        }
        ids
    };

    {
        let registry = app.state::<TaskRegistry>();
        let mut guard = registry.0.lock().unwrap();
        for task_id in &task_ids {
            if let Some(mut child) = guard.remove(task_id) {
                kill_child(&mut child);
            }
        }
    }

    let affected = app
        .state::<Db>()
        .0
        .lock()
        .unwrap()
        .execute("DELETE FROM projects WHERE id = ?1", [&project_id])
        .map_err(|error| error.to_string())?;
    if affected == 0 {
        return Err("项目不存在".into());
    }

    let project_dir = project_dir(&app, &project_id)?;
    if project_dir.exists() {
        let mut attempts = 0;
        loop {
            match fs::remove_dir_all(&project_dir) {
                Ok(()) => break,
                Err(_) if attempts < 3 => {
                    attempts += 1;
                    thread::sleep(std::time::Duration::from_millis(120 * attempts));
                }
                Err(error) => return Err(format!("删除项目目录失败：{error}")),
            }
        }
    }
    Ok(())
}

fn is_supported_image(path: &Path) -> bool {
    const SUPPORTED: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "gif"];
    path.extension()
        .map(|ext| ext.to_string_lossy().to_lowercase())
        .map(|ext| SUPPORTED.contains(&ext.as_str()))
        .unwrap_or(false)
}

#[tauri::command]
fn import_images(
    db: State<'_, Db>,
    project_id: String,
    paths: Vec<String>,
) -> Result<Vec<ImageRecord>, String> {
    let mut conn = db.0.lock().unwrap();
    let project_exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM projects WHERE id = ?1)",
            [&project_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if !project_exists {
        return Err("项目不存在".into());
    }

    let mut existing = HashSet::new();
    {
        let mut stmt = conn
            .prepare("SELECT path FROM images WHERE project_id = ?1")
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map([&project_id], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?;
        for row in rows {
            existing.insert(row.map_err(|error| error.to_string())?.to_lowercase());
        }
    }

    let mut order: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM images WHERE project_id = ?1",
            [&project_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;

    let tx = conn.transaction().map_err(|error| error.to_string())?;
    let mut imported = Vec::new();
    let now = now_millis();

    for raw in paths {
        let path = PathBuf::from(raw);
        if !is_supported_image(&path) {
            continue;
        }
        let normalized = path.to_string_lossy();
        let key = normalized.to_lowercase();
        if existing.contains(&key) {
            continue;
        }
        let meta = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(_) => continue,
        };
        if !meta.is_file() {
            continue;
        }

        let id = uuid::Uuid::new_v4().to_string();
        let filename = path
            .file_name()
            .map(|value| value.to_string_lossy().into_owned());
        let size = meta.len() as i64;
        tx.execute(
            "INSERT INTO images (id, project_id, path, filename, size, status, sort_order)
             VALUES (?1, ?2, ?3, ?4, ?5, 'pending', ?6)",
            params![id, project_id, normalized, filename, size, order],
        )
        .map_err(|error| error.to_string())?;

        existing.insert(key);
        order += 1;
        imported.push(ImageRecord {
            id,
            project_id: project_id.clone(),
            path: normalized.into_owned(),
            filename,
            width: None,
            height: None,
            size: Some(size),
            status: "pending".into(),
            thumbnail_path: None,
            sort_order: order - 1,
        });
    }

    tx.commit().map_err(|error| error.to_string())?;
    if !imported.is_empty() {
        conn.execute(
            "UPDATE projects SET updated_at = ?1 WHERE id = ?2",
            params![now, project_id],
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(imported)
}

#[tauri::command]
fn list_images(db: State<'_, Db>, project_id: String) -> Result<Vec<ImageRecord>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT
               id,
               project_id,
               path,
               filename,
               width,
               height,
               size,
               status,
               thumbnail_path,
               sort_order
             FROM images
             WHERE project_id = ?1
             ORDER BY sort_order",
        )
        .map_err(|error| error.to_string())?;
    let rows = stmt
        .query_map([&project_id], |row| {
            Ok(ImageRecord {
                id: row.get(0)?,
                project_id: row.get(1)?,
                path: row.get(2)?,
                filename: row.get(3)?,
                width: row.get(4)?,
                height: row.get(5)?,
                size: row.get(6)?,
                status: row.get(7)?,
                thumbnail_path: row.get(8)?,
                sort_order: row.get(9)?,
            })
        })
        .map_err(|error| error.to_string())?;

    let mut images = Vec::new();
    for row in rows {
        images.push(row.map_err(|error| error.to_string())?);
    }
    Ok(images)
}

fn update_task_from_payload(app: &tauri::AppHandle, task_id: &str, payload: &Value) {
    let status = payload
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or("running")
        .to_string();
    let stage = payload
        .get("stage")
        .and_then(|value| value.as_str())
        .unwrap_or("running")
        .to_string();
    let progress = payload
        .get("progress")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let error = payload
        .get("error")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    let now = now_millis();
    let project_status = match status.as_str() {
        "done" => "done",
        "failed" | "cancelled" => "idle",
        _ => "active",
    };
    let latest = payload
        .get("metrics")
        .and_then(|value| value.as_array())
        .and_then(|items| items.last())
        .cloned()
        .or_else(|| {
            payload.get("loss").or_else(|| payload.get("psnr")).and_then(|_| {
                let mut item = payload.clone();
                item.as_object_mut().map(|map| {
                    map.remove("taskId");
                    map.remove("status");
                    map.remove("stage");
                    map.remove("progress");
                    map.remove("metrics");
                });
                Some(item)
            })
        });

    let db = app.state::<Db>();
    let conn = db.0.lock().unwrap();
    let _ = conn.execute(
        "UPDATE tasks
         SET status = ?1,
           stage = ?2,
           progress = ?3,
           finished_at = CASE WHEN ?1 IN ('done', 'failed', 'cancelled') THEN ?4 ELSE finished_at END,
           error = ?5,
           latest_json = ?6
         WHERE id = ?7",
        params![
            status,
            stage,
            progress,
            now,
            error,
            latest.as_ref().map(|value| value.to_string()),
            task_id
        ],
    );
    let _ = conn.execute(
        "UPDATE projects
         SET status = ?1, updated_at = ?2
         WHERE id = (SELECT project_id FROM tasks WHERE id = ?3)",
        params![project_status, now, task_id],
    );

    if let Some(items) = payload.get("metrics").and_then(|value| value.as_array()) {
        let stmt = conn
            .prepare(
                "INSERT OR IGNORE INTO metrics
                 (id, task_id, step, psnr, ssim, lpips, loss, vram_mb, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .ok();
        if let Some(mut stmt) = stmt {
            for item in items {
                let Some(step) = item.get("step").and_then(|value| value.as_i64()) else {
                    continue;
                };
                let id = format!("{task_id}-{step}");
                let _ = stmt.execute(params![
                    id,
                    task_id,
                    step,
                    item.get("psnr").and_then(|value| value.as_f64()),
                    item.get("ssim").and_then(|value| value.as_f64()),
                    item.get("lpips").and_then(|value| value.as_f64()),
                    item.get("loss").and_then(|value| value.as_f64()),
                    item.get("vramMB").and_then(|value| value.as_i64()),
                    now,
                ]);
            }
        }
    }
}

fn mark_task_failed(app: &tauri::AppHandle, task_id: &str, message: &str) {
    let payload = json!({
        "taskId": task_id,
        "status": "failed",
        "stage": "error",
        "progress": 0.0,
        "error": message,
    });
    update_task_from_payload(app, task_id, &payload);
    let _ = app.emit("titan://task", payload);
}

fn spawn_task_watcher(app: tauri::AppHandle, task_id: String, project_dir: PathBuf) {
    thread::spawn(move || {
        let status_path = project_dir.join("status.json");
        let mut last_raw: Option<String> = None;
        loop {
            if let Ok(raw) = fs::read_to_string(&status_path) {
                if last_raw.as_deref() != Some(raw.as_str()) {
                    if let Ok(payload) = serde_json::from_str::<Value>(&raw) {
                        last_raw = Some(raw);
                        update_task_from_payload(&app, &task_id, &payload);
                        let _ = app.emit("titan://task", payload);
                    }
                }
            }

            let exited = {
                let registry = app.state::<TaskRegistry>();
                let mut guard = registry.0.lock().unwrap();
                match guard.get_mut(&task_id) {
                    Some(child) => child
                        .try_wait()
                        .map(|status| status.is_some())
                        .unwrap_or(true),
                    None => true,
                }
            };

            if !exited {
                thread::sleep(std::time::Duration::from_millis(400));
                continue;
            }

            let final_payload: Option<Value> = fs::read_to_string(&status_path)
                .ok()
                .and_then(|raw| serde_json::from_str(&raw).ok());
            let terminal = final_payload
                .as_ref()
                .and_then(|payload| payload.get("status"))
                .and_then(|value| value.as_str())
                .map(|status| matches!(status, "done" | "failed" | "cancelled"))
                .unwrap_or(false);

            if let Some(payload) = final_payload {
                update_task_from_payload(&app, &task_id, &payload);
                let _ = app.emit("titan://task", payload);
            }

            if !terminal {
                let db = app.state::<Db>();
                let db_status: Option<String> = db
                    .0
                    .lock()
                    .unwrap()
                    .query_row(
                        "SELECT status FROM tasks WHERE id = ?1",
                        [&task_id],
                        |row| row.get(0),
                    )
                    .ok();
                if db_status.as_deref() == Some("cancelled") {
                    let payload = json!({
                        "taskId": task_id,
                        "status": "cancelled",
                        "stage": "cancel",
                        "progress": 0.0,
                        "error": "用户取消",
                    });
                    update_task_from_payload(&app, &task_id, &payload);
                    let _ = app.emit("titan://task", payload);
                } else {
                    mark_task_failed(&app, &task_id, "任务被停止或引擎异常退出");
                }
            }

            app.state::<TaskRegistry>()
                .0
                .lock()
                .unwrap()
                .remove(&task_id);
            break;
        }
    });
}

#[tauri::command]
fn get_task(db: State<'_, Db>, project_id: String) -> Result<Option<TaskInfo>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT
               id,
               project_id,
               type,
               status,
               stage,
               progress,
               started_at,
               finished_at,
               error,
               latest_json
             FROM tasks
             WHERE project_id = ?1
             ORDER BY started_at DESC
             LIMIT 1",
        )
        .map_err(|error| error.to_string())?;
    let rows = stmt
        .query_map([&project_id], |row| {
            Ok(TaskInfo {
                id: row.get(0)?,
                project_id: row.get(1)?,
                kind: row.get(2)?,
                status: row.get(3)?,
                stage: row.get(4)?,
                progress: row.get(5)?,
                started_at: row.get(6)?,
                finished_at: row.get(7)?,
                error: row.get(8)?,
                latest: row
                    .get::<_, Option<String>>(9)?
                    .and_then(|raw| serde_json::from_str(&raw).ok()),
            })
        })
        .map_err(|error| error.to_string())?;
    let mut task = rows
        .into_iter()
        .next()
        .transpose()
        .map_err(|error| error.to_string())?;
    if let Some(ref mut current) = task {
        if current.latest.is_none() {
            current.latest = conn
                .query_row(
                    "SELECT step, psnr, ssim, loss, vram_mb
                     FROM metrics
                     WHERE task_id = ?1
                     ORDER BY step DESC
                     LIMIT 1",
                    [&current.id],
                    |row| {
                        Ok(json!({
                            "step": row.get::<_, Option<i64>>(0)?,
                            "psnr": row.get::<_, Option<f64>>(1)?,
                            "ssim": row.get::<_, Option<f64>>(2)?,
                            "loss": row.get::<_, Option<f64>>(3)?,
                            "vramMB": row.get::<_, Option<i64>>(4)?,
                        }))
                    },
                )
                .ok();
        }
    }
    Ok(task)
}

#[tauri::command]
fn start_reconstruction(
    app: tauri::AppHandle,
    db: State<'_, Db>,
    project_id: String,
    config: Option<serde_json::Value>,
) -> Result<TaskInfo, String> {
    let conn = db.0.lock().unwrap();
    let image_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM images WHERE project_id = ?1",
            [&project_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if image_count == 0 {
        return Err("请先导入图片再开始重建".into());
    }

    let project_dir = project_dir(&app, &project_id)?;
    fs::create_dir_all(&project_dir).map_err(|error| error.to_string())?;

    let mut paths = Vec::new();
    {
        let mut stmt = conn
            .prepare("SELECT path FROM images WHERE project_id = ?1 ORDER BY sort_order")
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map([&project_id], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?;
        for row in rows {
            paths.push(row.map_err(|error| error.to_string())?);
        }
    }
    let inputs = json!({ "images": paths });
    fs::write(
        project_dir.join("inputs.json"),
        serde_json::to_string_pretty(&inputs).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    let mut merged = json!({
        "engine": "facets",
        "resolution": 192,
        "steps": 1200,
        "train_views": 16,
        "patch_size": 1024,
        "eval_every": 100,
        "learning_rate": 0.02,
        "gpu_budget": 8,
        "render_batch_size": 256,
        "max_splats": 12000,
        "global_residual_resolution": 16,
        "global_residual_weight": 0.6,
        "init_target_px": 1.3,
        "sparse_points_limit": 12000,
        "background_residual_weight": 0.0,
        "disk_cutoff": 1.8,
        "falloff_power": 1.0,
        "pose_source": "auto",
        "colmap_path": "",
        "sfm_matcher": "sequential",
        "sfm_max_image_size": 1600,
        "sfm_min_images": 4,
        "sfm_min_registered": 4,
        "sfm_use_gpu": true
    });
    if let Some(user_config) = config {
        if let Some(object) = user_config.as_object() {
            for (key, value) in object {
                merged[key] = value.clone();
            }
        }
    }

    let engine_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("无法定位引擎目录")?
        .join("python");
    if !engine_root.join("titan_engine").join("main.py").exists() {
        return Err(format!("未找到 Python 引擎：{}", engine_root.display()));
    }
    let python = resolve_python()?;

    let task_id = uuid::Uuid::new_v4().to_string();
    let now = now_millis();
    let config_raw = merged.to_string();
    conn.execute(
        "INSERT INTO tasks
           (id, project_id, type, status, stage, progress, started_at, config_json)
         VALUES (?1, ?2, 'reconstruction', 'running', 'data', 0.01, ?3, ?4)",
        params![task_id, project_id, now, config_raw],
    )
    .map_err(|error| error.to_string())?;
    conn.execute(
        "UPDATE projects SET status = 'active', updated_at = ?1 WHERE id = ?2",
        params![now, project_id],
    )
    .map_err(|error| error.to_string())?;

    let config_arg = merged.to_string();
    let project_arg = project_dir.to_string_lossy().into_owned();
    let task_arg = task_id.clone();
    let stdout_file = fs::File::create(project_dir.join("engine.log"))
        .map_err(|error| format!("无法创建引擎日志：{error}"))?;
    let stderr_file = fs::File::create(project_dir.join("engine.error.log"))
        .map_err(|error| format!("无法创建引擎错误日志：{error}"))?;
    let mut command = Command::new(&python);
    command
        .arg("-m")
        .arg("titan_engine.main")
        .arg("--project-dir")
        .arg(&project_arg)
        .arg("--task-id")
        .arg(&task_arg)
        .arg("--config-json")
        .arg(&config_arg)
        .current_dir(&engine_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));
    #[cfg(windows)]
    command.creation_flags(0x08000000);

    let child = command.spawn().map_err(|error| {
        let _ = conn.execute(
            "UPDATE tasks SET status = 'failed', error = ?1 WHERE id = ?2",
            params![format!("无法启动引擎：{error}"), task_id],
        );
        format!("无法启动 Python 引擎：{error}")
    })?;
    app.state::<TaskRegistry>()
        .0
        .lock()
        .unwrap()
        .insert(task_id.clone(), child);

    drop(conn);
    spawn_task_watcher(app.clone(), task_id.clone(), project_dir);

    Ok(TaskInfo {
        id: task_id,
        project_id,
        kind: "reconstruction".into(),
        status: "running".into(),
        stage: "data".into(),
        progress: 0.01,
        started_at: Some(now),
        finished_at: None,
        error: None,
        latest: None,
    })
}

#[tauri::command]
fn pause_task(app: tauri::AppHandle, task_id: String) -> Result<(), String> {
    let project_id = task_project_id(&app, &task_id)?;
    let project_dir = project_dir(&app, &project_id)?;
    write_control(&project_dir, "pause")?;
    app.state::<Db>()
        .0
        .lock()
        .unwrap()
        .execute(
            "UPDATE tasks SET status = 'paused' WHERE id = ?1",
            [&task_id],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
fn resume_task(app: tauri::AppHandle, task_id: String) -> Result<(), String> {
    let project_id = task_project_id(&app, &task_id)?;
    let project_dir = project_dir(&app, &project_id)?;
    write_control(&project_dir, "resume")?;
    app.state::<Db>()
        .0
        .lock()
        .unwrap()
        .execute(
            "UPDATE tasks SET status = 'running' WHERE id = ?1",
            [&task_id],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
fn cancel_task(app: tauri::AppHandle, task_id: String) -> Result<(), String> {
    let project_id = task_project_id(&app, &task_id)?;
    let project_dir = project_dir(&app, &project_id)?;
    write_control(&project_dir, "cancel")?;
    let child = app
        .state::<TaskRegistry>()
        .0
        .lock()
        .unwrap()
        .remove(&task_id);
    if let Some(mut child) = child {
        kill_child(&mut child);
    }
    app.state::<Db>()
        .0
        .lock()
        .unwrap()
        .execute(
            "UPDATE tasks
             SET status = 'cancelled', finished_at = ?1, error = '用户取消'
             WHERE id = ?2",
            params![now_millis(), task_id],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
fn get_metrics(db: State<'_, Db>, task_id: String) -> Result<Vec<Value>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT step, psnr, ssim, lpips, loss, vram_mb, created_at
             FROM metrics
             WHERE task_id = ?1
             ORDER BY step",
        )
        .map_err(|error| error.to_string())?;
    let rows = stmt
        .query_map([&task_id], |row| {
            Ok(json!({
                "step": row.get::<_, i64>(0)?,
                "psnr": row.get::<_, Option<f64>>(1)?,
                "ssim": row.get::<_, Option<f64>>(2)?,
                "lpips": row.get::<_, Option<f64>>(3)?,
                "loss": row.get::<_, Option<f64>>(4)?,
                "vramMB": row.get::<_, Option<i64>>(5)?,
                "createdAt": row.get::<_, i64>(6)?,
            }))
        })
        .map_err(|error| error.to_string())?;
    let mut metrics = Vec::new();
    for row in rows {
        metrics.push(row.map_err(|error| error.to_string())?);
    }
    Ok(metrics)
}

#[tauri::command]
fn get_scene(app: tauri::AppHandle, project_id: String) -> Result<Option<Value>, String> {
    let scene_path = project_dir(&app, &project_id)?.join("scene.json");
    if !scene_path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&scene_path).map_err(|error| error.to_string())?;
    serde_json::from_str(&raw).map(Some).map_err(|error| error.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let db_path = app
                .path()
                .app_data_dir()
                .map_err(|error| error.to_string())?
                .join("titan.db");
            let db = Db::open(&db_path)?;
            app.manage(db);
            app.manage(TaskRegistry(Mutex::new(HashMap::new())));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_info,
            create_project,
            list_projects,
            delete_project,
            import_images,
            list_images,
            start_reconstruction,
            pause_task,
            resume_task,
            cancel_task,
            get_task,
            get_metrics,
            get_scene
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_upgrades_legacy_task_schema() {
        let path = std::env::temp_dir().join(format!(
            "titan-legacy-{}.db",
            uuid::Uuid::new_v4()
        ));
        {
            let conn = Connection::open(&path).expect("open legacy db");
            conn.execute_batch(
                "CREATE TABLE tasks (
                   id TEXT PRIMARY KEY,
                   project_id TEXT NOT NULL,
                   type TEXT NOT NULL,
                   status TEXT NOT NULL
                 );
                 INSERT INTO tasks (id, project_id, type, status)
                 VALUES ('t1', 'p1', 'reconstruction', 'running');",
            )
            .expect("create legacy tasks");
        }

        let db = Db::open(&path).expect("migrate legacy db");
        let conn = db.0.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT stage, progress, config_json, latest_json
                 FROM tasks WHERE id = ?1",
                ["t1"],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, Option<String>>(3)?,
                    ))
                },
            )
            .expect("query migrated columns");

        assert_eq!(row.0, "data");
        assert_eq!(row.1, 0.0);
        assert!(row.2.is_none());
        assert!(row.3.is_none());

        drop(conn);
        drop(db);
        let _ = fs::remove_file(&path);
    }
}
