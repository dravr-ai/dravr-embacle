// ABOUTME: GET /v1/uhp — the unauthenticated protocol discovery document
// ABOUTME: Capabilities are reported honestly; conformance_class is derived from them
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Discovery.
//!
//! Served without a credential on purpose: a client must be able to learn
//! whether this is a UHP server, and which versions it speaks, *before*
//! deciding what to present. The document holds nothing principal-specific.
//!
//! Capabilities are named booleans and are reported as they actually are. The
//! specification is explicit that a server reports `false` for something it
//! does not implement rather than omitting it, so a client can tell "not
//! supported" from "server predates this field".

use std::collections::BTreeMap;

use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;

use super::version::{SUPPORTED, VERSION};

/// What this server can actually do, as named booleans.
///
/// The specification calls for named booleans and permits any additional key,
/// so this is carried as an ordered map rather than a struct of `bool` fields:
/// it serialises to exactly the same JSON, keeps the capability names in one
/// list beside the class rule that reads them, and lets a future capability be
/// added without reshaping a type.
///
/// A capability this server does not implement is reported `false`, never
/// omitted — the specification is explicit that a client must be able to tell
/// "not supported" from "server predates this field".
#[derive(Debug, Clone, Serialize)]
pub struct Capabilities(BTreeMap<&'static str, bool>);

/// The capabilities `core` requires, all of which must be true to claim it.
const CORE: [&str; 3] = ["streaming", "sessions", "cancellation"];

/// What `extended` adds on top of [`CORE`].
const EXTENDED: [&str; 3] = ["files_input", "files_output", "session_listing"];

/// What `full` adds on top of [`EXTENDED`].
const FULL: [&str; 1] = ["harness_management"];

/// Every capability name the specification defines, in document order.
const ALL: [&str; 9] = [
    "streaming",
    "sessions",
    "cancellation",
    "files_input",
    "files_output",
    "session_listing",
    "harness_management",
    "session_sharing",
    "idempotency",
];

impl Capabilities {
    /// What this build implements today.
    ///
    /// Every flag here is the truth, so a client planning against this
    /// document is never told a surface exists that would fail on first use.
    /// A capability is added to this list when its conformance checks pass,
    /// not when its handler exists.
    pub fn current() -> Self {
        let implemented = [
            "streaming",
            "sessions",
            "cancellation",
            "session_listing",
            "harness_management",
            "session_sharing",
            "files_input",
            "files_output",
        ];
        Self(
            ALL.iter()
                .map(|&name| (name, implemented.contains(&name)))
                .collect(),
        )
    }

    /// Whether every named capability is present and true.
    fn all_of(&self, names: &[&str]) -> bool {
        names
            .iter()
            .all(|n| self.0.get(n).copied().unwrap_or(false))
    }

    /// The highest class these capabilities actually satisfy.
    ///
    /// The enum admits only `core`, `extended` and `full`, so a server that
    /// does not yet meet `core` still has to name one. It names `core` — the
    /// floor it is building toward — and the shortfall surfaces as a failing
    /// D-05 rather than as a document that quietly overstates itself.
    pub fn class(&self) -> &'static str {
        if !self.all_of(&CORE) {
            return "core";
        }
        if !self.all_of(&EXTENDED) {
            return "core";
        }
        if self.all_of(&FULL) {
            "full"
        } else {
            "extended"
        }
    }
}

/// The discovery document.
#[derive(Debug, Clone, Serialize)]
pub struct Discovery {
    /// Always `uhp.discovery`.
    pub object: &'static str,
    /// Always `uhp`.
    pub protocol: &'static str,
    /// Every version this server can serve.
    pub versions: Vec<&'static str>,
    /// The version served when a client asks for none.
    pub default_version: &'static str,
    /// The class these capabilities satisfy.
    pub conformance_class: &'static str,
    /// Named booleans.
    pub capabilities: Capabilities,
    /// Who is serving.
    pub implementation: Implementation,
}

/// Names the server behind the protocol.
#[derive(Debug, Clone, Serialize)]
pub struct Implementation {
    /// Crate name.
    pub name: &'static str,
    /// Crate version.
    pub version: &'static str,
}

/// Handle `GET /v1/uhp`.
pub async fn handle() -> impl IntoResponse {
    let capabilities = Capabilities::current();
    let conformance_class = capabilities.class();
    Json(Discovery {
        object: "uhp.discovery",
        protocol: "uhp",
        versions: SUPPORTED.to_vec(),
        default_version: VERSION,
        conformance_class,
        capabilities,
        implementation: Implementation {
            name: "embacle-server",
            version: env!("CARGO_PKG_VERSION"),
        },
    })
}
