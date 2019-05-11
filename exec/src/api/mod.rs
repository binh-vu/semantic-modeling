use zmq;
use serde_json;
mod in_mem_db;
mod service;

pub mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain!{
        foreign_links {
            FromUtf8Error(::std::string::FromUtf8Error);
        }
    }
}

use self::errors::*;

pub fn server_wrapper() {
    if let Err(ref e) = server() {
        println!("error: {}", e);
        for e in e.iter().skip(1) {
            println!("caused by: {}", e);
        }

        // The backtrace is not always generated. Try to run this example
        // with `RUST_BACKTRACE=1`.
        if let Some(backtrace) = e.backtrace() {
            println!("{:?}", backtrace);
        }

        ::std::process::exit(1);
    }
}

fn server() -> errors::Result<()> {
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::REP).chain_err(|| "Cannot create ZeroMQ socket")?;
    socket.bind("ipc:///tmp/ruek-response.ipc").chain_err(|| "Cannot bind to socket")?;

    let receive_message = || {
        socket.recv_bytes(0).map(|bytes| String::from_utf8(bytes)).chain_err(|| "Invalid utf8 bytes")
    };
    let send_success_resp = |result: &serde_json::Value| {
        socket.send_str(&success_resp(result), 0).chain_err(|| "Cannot send response")
    };
    let send_error_resp = |error: &errors::Error| {
        socket.send_str(&error_resp(error), 0).chain_err(|| "Cannot send response")
    };

    // storage
    let mut in_mem_db = in_mem_db::InMemoryDB::new();

    // handle logic here
    loop {
        let raw_command: String = receive_message()??;
        match service::execute_command_with_args(&raw_command, &mut in_mem_db) {
            Ok(x) => send_success_resp(&x)?,
            Err(e) => send_error_resp(&e)?
        }
    }
}

fn success_resp(result: &serde_json::Value) -> String {
    serde_json::to_string(&json!({
        "status": "success",
        "result": result
    })).expect("Resp must be JSON serializable")
}

fn error_resp(error: &errors::Error) -> String {
    let mut error_message = format!("{}\n", error);
    for e in error.iter().skip(1) {
        error_message.push_str(&format!("caused by: {}\n", e));
    }

    serde_json::to_string(&json!({
        "status": "error",
        "error": error_message,
    })).expect("Resp must be JSON serializable")
}