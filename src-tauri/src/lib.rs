use rusqlite::{params, Connection};
use serde::Serialize;
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::{Manager, State};

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
              progress REAL NOT NULL DEFAULT 0,
              started_at INTEGER,
              finished_at INTEGER,
              error TEXT
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
        Ok(Self(Mutex::new(conn)))
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
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
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_info,
            create_project,
            list_projects,
            import_images,
            list_images
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
