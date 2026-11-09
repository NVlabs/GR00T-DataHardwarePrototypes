# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope existing view assertions to a section of the combined document."""

from html.parser import HTMLParser


class Section(HTMLParser):
    def __init__(self, section_id):
        super().__init__(convert_charrefs=False)
        self.section_id = section_id
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if not self.depth and dict(attrs).get("id") == self.section_id:
            self.depth = 1
        elif self.depth and tag == "section":
            self.depth += 1
        if self.depth:
            self.parts.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        if self.depth:
            self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if self.depth:
            self.parts.append(f"</{tag}>")
            if tag == "section":
                self.depth -= 1

    def handle_data(self, value):
        if self.depth:
            self.parts.append(value)

    def handle_entityref(self, name):
        self.handle_data(f"&{name};")

    def handle_charref(self, name):
        self.handle_data(f"&#{name};")


def section_html(html, section_id):
    section = Section(section_id)
    section.feed(html)
    return "".join(section.parts) if section.parts else html
