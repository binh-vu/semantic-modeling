use std::process::Command;
use std::path::*;
use algorithm::data_structure::graph::*;
use serde_json;
use std::fs::*;

pub fn clear_dir() {
    let path = Path::new("/tmp/sm_debugging");
    if path.exists() {
        remove_dir_all(&path).unwrap();
    }
}

pub fn draw_graphs_same_dir(graphs: &[Graph]) {
    let foutput = Path::new("/tmp/sm_debugging/draw_graphs.json");
    if !foutput.parent().unwrap().exists() {
        create_dir(foutput.parent().unwrap()).unwrap();
    }

    let obj = json!({
        "graphs": graphs,
    });
    serde_json::to_writer(File::create(foutput).unwrap(), &obj).unwrap();

    let mut py_code = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    py_code.push("src/debug_utils.py");

    let output = Command::new("python")
        .arg(py_code.as_os_str())
        .arg("draw_graph")
        .arg("true")
        .output()
        .expect("Fail to execute python process");
    assert!(output.status.success(), format!("{:?}", output));
}

pub fn draw_graphs(graphs: &[Graph]) {
    let foutput = Path::new("/tmp/sm_debugging/draw_graphs.json");
    if !foutput.parent().unwrap().exists() {
        create_dir(foutput.parent().unwrap()).unwrap();
    }

    let obj = json!({
        "graphs": graphs,
    });
    serde_json::to_writer(File::create(foutput).unwrap(), &obj).unwrap();

    let mut py_code = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    py_code.push("src/debug_utils.py");

    let output = Command::new("python")
        .arg(py_code.as_os_str())
        .arg("draw_graph")
        .arg("false")
        .output()
        .expect("Fail to execute python process");
    assert!(output.status.success(), format!("{:?}", output));
}