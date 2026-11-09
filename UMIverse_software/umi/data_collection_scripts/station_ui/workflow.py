# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the existing workflow views into one document, keeping their actions."""

import json

from jinja2 import pass_context
from markupsafe import Markup

from station_ui import readiness, recording, tokens
from station_ui.rig import claim, is_owner

SECTIONS = (
    ("setup", "Setup"),
    ("configuration", "Configuration"),
    ("calibration", "Calibration"),
    ("record", "Record"),
    ("audit", "Data Audit"),
)


@pass_context
def render_section(context, section):
    values = dict(context)
    values.update(section["context"])
    values["embedded"] = True
    return Markup(context.environment.get_template(section["template"]).render(values))


def progress_header(request, token):
    return json.dumps(
        {
            "done": sorted(readiness.done(request.app)),
            "locked": recording.in_progress(request.app),
            "owner": is_owner(request.app.state.rig, token),
        }
    )


def render_page(request, current, context, *, token=None, status_code=200):
    from station_ui.routes import audit, calibration, config, record, setup

    if token is None and request.method == "GET":
        token = tokens.new_token()
        claim(request.app.state.rig, token)

    def configuration_context():
        state = request.app.state
        values = config._page_context(
            request,
            token,
            notice=getattr(state, "config_notice", ""),
            warnings=getattr(state, "config_warnings", []),
        )
        # A calibration/record poll selects a small fragment out of this page;
        # it must not consume notices the operator never receives.
        if not request.headers.get("HX-Request"):
            state.config_notice = ""
            state.config_warnings = []
        return values

    builders = {
        "setup": lambda: setup._page_context(request, token),
        "config": configuration_context,
        "calibration": lambda: calibration._page_context(request),
        "record": lambda: record._context(request, token),
        "audit": lambda: audit._page_context(request, token),
    }
    sections = []
    for key, label in SECTIONS:
        source = "config" if key == "configuration" else key
        values = context if source == current else builders[source]()
        values = {**values, "can_edit": is_owner(request.app.state.rig, token)}
        sections.append(
            {
                "id": key,
                "label": label,
                "template": f"{source}.html",
                "context": values,
                "href": "/" if source == "setup" else "/" + source,
            }
        )
    progress = json.loads(progress_header(request, token))
    response = request.app.state.templates.TemplateResponse(
        request,
        "workflow.html",
        {
            "page_title": "Data collection",
            "active": "/" + current,
            "workflow_sections": sections,
            "workflow": True,
            "workflow_progress": progress,
            "can_edit": is_owner(request.app.state.rig, token),
        },
        status_code=status_code,
    )
    response.headers["X-Workflow-Progress"] = json.dumps(progress)
    if token:
        tokens.remember(response, token)
    return response
