/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Update assignment diagrams from current fields without replacing inputs. Apply and
   Record validate the same USB ownership evidence. */
(() => {
  "use strict";

  const kinds = {gripper: "gripper", umi: "oak", tracking: "iphone"};
  const labels = {gripper: "Gripper", umi: "Top camera", tracking: "Phone"};

  function check(values, inventory) {
    const devices = {}, owners = {}, affected = new Set(), messages = [];
    for (const [role, kind] of Object.entries(kinds)) {
      const value = (values[role] || "").trim();
      devices[role] = inventory.aliases[kind][value.toLowerCase()] || value;
      owners[role] = inventory.owners[kind][devices[role]];
    }
    for (const [first, second] of [["umi", "tracking"], ["gripper", "umi"], ["gripper", "tracking"]]) {
      if (owners[first] && owners[second] && owners[first] !== owners[second]) {
        affected.add(first);
        affected.add(second);
        messages.push(`${labels[first]} ${devices[first]} belongs to ${owners[first]}; ${labels[second].toLowerCase()} ${devices[second]} belongs to ${owners[second]}.`);
      }
    }
    if (affected.size) {
      const remedy = "tracking" in values ? " Select the phone and top camera from the same gripper row in Detected devices." : " Select the top camera from the selected gripper's row in Detected devices.";
      return {state: "mismatch", affected: [...affected].sort(), message: "Pairing mismatch. " + messages[0] + remedy};
    }
    const required = Object.keys(kinds).filter(role => role in values);
    if (required.some(role => !values[role].trim())) {
      return {state: "incomplete", affected: [], message: "Choose the devices below to check their pairing."};
    }
    if (required.length && required.every(role => owners[role])) {
      return {state: "matched", affected: [], message: "Pairing verified for this gripper's top camera" + ("tracking" in values ? " and phone." : ".")};
    }
    return {state: "unverified", affected: [], message: "Pairing not verified. Connect the devices through the gripper's USB hub, then Refresh devices."};
  }


  if (typeof module !== "undefined" && module.exports) module.exports = {check};
  if (typeof document === "undefined") return;

  function selected(input, names) {
    const value = input.value.trim();
    // Resolve unique suffixes but never claim a pairing from ambiguous partial text.

    const options = names || (input.list ? Array.from(input.list.options, option => option.value) : []);
    const exact = options.find(option => option.toLowerCase() === value.toLowerCase());
    if (exact) return exact;
    if (!value) return "";
    const matches = options.filter(option => {
      if (/^\d+$/.test(value)) {
        const trailing = option.match(/(\d+)$/);
        return trailing && Number(trailing[1]) === Number(value);
      }
      return option.toLowerCase().endsWith(value.toLowerCase());
    });
    return matches.length === 1 ? matches[0] : value;
  }

  function refresh() {
    const panel = document.getElementById("gripper-diagrams");
    const data = panel?.querySelector("[data-gripper-evidence]");
    if (!data) return;
    const inventory = JSON.parse(data.textContent);
    document.querySelectorAll("[data-gripper-diagram]").forEach(diagram => {
      const values = {};
      const assignments = diagram.querySelectorAll("[data-gripper-selection]");
      assignments.forEach(assignment => {
        const input = document.getElementById(assignment.dataset.field);
        const role = assignment.dataset.gripperSelection;
        values[role] = input ? selected(input, inventory.options?.[assignment.dataset.kind]) : "";
        const label = assignment.querySelector("[data-gripper-value]");
        const text = values[role] || "Not selected";
        if (label.textContent !== text) label.textContent = text;
      });
      const result = check(values, inventory);
      diagram.dataset.pairingState = result.state;
      diagram.querySelectorAll("[data-gripper-part]").forEach(part => {
        part.classList.toggle("is-mismatch", result.affected.includes(part.dataset.gripperPart));
      });
      assignments.forEach(assignment => {
        const bad = result.affected.includes(assignment.dataset.gripperSelection);
        assignment.classList.toggle("is-mismatch", bad);
        assignment.querySelector("[data-gripper-flag]").hidden = !bad;
        const input = document.getElementById(assignment.dataset.field);
        if (!input) return;
        input.classList.toggle("input--pairing-mismatch", bad);
        // Clear stale Apply warnings when current assignment evidence supersedes them.

        const field = input.closest("[data-device-field]");
        const error = field?.querySelector("[data-pairing-error]");
        if (error) {
          error.hidden = true;
          input.removeAttribute("aria-invalid");
        }
        const description = new Set((input.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean));
        if (error) description.delete(error.id);
        const statusId = `pairing-${diagram.dataset.gripperDiagram}`;
        if (bad) description.add(statusId);
        else description.delete(statusId);
        if (description.size) input.setAttribute("aria-describedby", [...description].join(" "));
        else input.removeAttribute("aria-describedby");
        if (bad) {
          if (!input.hasAttribute("data-pairing-invalid")) input.dataset.pairingInvalid = input.getAttribute("aria-invalid") || "";
          input.setAttribute("aria-invalid", "true");
        } else if (input.hasAttribute("data-pairing-invalid")) {
          if (input.dataset.pairingInvalid) input.setAttribute("aria-invalid", input.dataset.pairingInvalid);
          else input.removeAttribute("aria-invalid");
          delete input.dataset.pairingInvalid;
        }
      });
      const status = diagram.querySelector("[data-gripper-status]");
      const text = (result.state === "mismatch" ? "⚠ " : "") + (result.state === "matched" ? "Pairing verified" : result.message);
      status.setAttribute("aria-label", result.message);
      if (status.textContent !== text) status.textContent = text;
    });
  }

  document.addEventListener("input", event => {
    if (event.target.closest("[data-device-field]")) refresh();
  });
  document.addEventListener("change", event => {
    if (event.target.closest("[data-device-field]")) refresh();
  });
  document.addEventListener("click", event => {
    const assignment = event.target.closest("[data-gripper-selection]");
    if (!assignment) return;
    const input = document.getElementById(assignment.dataset.field);
    if (input) {
      event.preventDefault();
      input.focus();
      input.scrollIntoView({block: "center", behavior: "smooth"});
    }
  });
  document.addEventListener("htmx:afterSwap", refresh);
  document.addEventListener("htmx:afterSettle", refresh);
  refresh();
})();
