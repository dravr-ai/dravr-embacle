// ABOUTME: A canned-HTTP listener on 127.0.0.1 for the HTTP provider suites: serves answers, records requests
// ABOUTME: Shared by the Anthropic suites so no request in them leaves the machine
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(clippy::expect_used, clippy::str_to_string)]
#![cfg(feature = "http-api")]

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

/// One request as the listener received it
pub struct Recorded {
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// A complete HTTP/1.1 answer that closes its connection
pub fn http(status: u16, body: &str) -> String {
    format!(
        "HTTP/1.1 {status} Canned\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// Read one request: headers up to the blank line, then `Content-Length` bytes
async fn read_request(stream: &mut TcpStream) -> Recorded {
    let mut raw = Vec::new();
    let mut buf = [0_u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut buf).await.expect("read request");
        assert!(n > 0, "connection closed before the headers ended");
        raw.extend_from_slice(&buf[..n]);
        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&raw[..header_end]).to_string();
    let mut lines = head.lines();
    let path = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or_default()
        .to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    let length: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    while raw.len() < header_end + length {
        let n = stream.read(&mut buf).await.expect("read body");
        assert!(n > 0, "connection closed before the body ended");
        raw.extend_from_slice(&buf[..n]);
    }
    Recorded {
        path,
        headers,
        body: String::from_utf8_lossy(&raw[header_end..header_end + length]).to_string(),
    }
}

/// Serve `answers` in order, one connection each, and return what was asked
pub async fn serve(answers: Vec<String>) -> (String, JoinHandle<Vec<Recorded>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    let handle = tokio::spawn(async move {
        let mut recorded = Vec::new();
        for answer in answers {
            let (mut stream, _) = listener.accept().await.expect("accept");
            recorded.push(read_request(&mut stream).await);
            stream.write_all(answer.as_bytes()).await.expect("write");
            stream.shutdown().await.ok();
        }
        recorded
    });
    (base, handle)
}
