use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AppInfo {
    name: &'static str,
    version: &'static str,
    engine: &'static str,
    gpu_budget: &'static str,
    target_platform: &'static str,
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![app_info])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
