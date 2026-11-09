/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Device dropdowns show the complete current datalist regardless of the typed value;
   typing retains filtering. Read options at open time so scans stay authoritative.
   Task search also offers new names. Without JavaScript, native inputs and datalists
   remain usable. */
(function () {
  var OPEN = "combo--open";
  var SEARCH = "combo--search";
  var open = null; // the one .combo showing a list, or null

  function datalistFor(input) {
    var id = input.getAttribute("list") || input.getAttribute("data-list");
    return id ? document.getElementById(id) : null;
  }

  function close() {
    if (!open) return;
    var list = open.querySelector(".combo__list");
    if (list) list.remove();
    open.classList.remove(OPEN);
    var toggle = open.querySelector(".combo__toggle");
    if (toggle) toggle.setAttribute("aria-expanded", "false");
    open = null;
  }

  /* Use aria-activedescendant to expose list selection while focus stays in the
     input. */
  function highlight(combo, index) {
    var rows = combo.querySelectorAll(".combo__option");
    if (!rows.length) return;
    var at = Math.max(0, Math.min(index, rows.length - 1));
    for (var i = 0; i < rows.length; i++) {
      rows[i].setAttribute("aria-selected", i === at ? "true" : "false");
    }
    rows[at].scrollIntoView({ block: "nearest" });
    var input = combo.querySelector(".input");
    if (input) input.setAttribute("aria-activedescendant", rows[at].id);
  }

  function selectedIndex(combo) {
    var rows = combo.querySelectorAll(".combo__option");
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].getAttribute("aria-selected") === "true") return i;
    }
    return -1;
  }

  /* Dispatch change after setting value programmatically so linked-camera and form
     updates run. */
  function choose(combo, value) {
    var input = combo.querySelector(".input");
    close();
    if (!input) return;
    input.value = value;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    input.focus();
  }

  /* `query` narrows the list to options whose value or label contains it; the
     ▾ passes none, which shows the full list. */
  function openList(combo, query) {
    var input = combo.querySelector(".input");
    if (!input || input.readOnly || input.disabled) return;
    var source = datalistFor(input);
    if (!source) return;

    close();

    var search = combo.classList.contains(SEARCH);
    var typed = (query || "").trim();
    var needle = typed.toLowerCase();
    var options = [];
    var exact = false;
    for (var k = 0; k < source.options.length; k++) {
      var candidate = source.options[k];
      if (candidate.value === typed) exact = true;
      var text = (candidate.value + " " + candidate.label).toLowerCase();
      if (!needle || text.indexOf(needle) !== -1) options.push(candidate);
    }
    var list = document.createElement("ul");
    list.className = "combo__list";
    list.id = combo.id + "-list";
    list.setAttribute("role", "listbox");

    /* Offer the typed task as new only when no existing option matches it exactly. */
    if (search && typed && !exact) {
      var add = document.createElement("li");
      add.className = "combo__option combo__option--new";
      add.id = list.id + "-new";
      add.setAttribute("role", "option");
      add.setAttribute("aria-selected", "false");
      add.setAttribute("data-value", typed);
      var action = document.createElement("span");
      action.className = "combo__action";
      action.textContent = "+ Add new task";
      add.appendChild(action);
      var named = document.createElement("span");
      named.className = "combo__value";
      named.textContent = typed;
      add.appendChild(named);
      list.appendChild(add);
    } else if (search && !options.length) {
      var none = document.createElement("li");
      none.className = "combo__empty";
      none.textContent = "No previous tasks yet — type a name to add one.";
      list.appendChild(none);
    }

    if (!search && !options.length) {
      /* Distinguish an empty result from a scan in progress; the latter should wait
         rather than ask for another refresh. */
      var empty = document.createElement("li");
      empty.className = "combo__empty";
      empty.textContent = source.hasAttribute("data-scanning")
        ? "Looking for devices…"
        : "No devices found — press Refresh devices.";
      list.appendChild(empty);
    }

    for (var i = 0; i < options.length; i++) {
      var option = options[i];
      var row = document.createElement("li");
      row.className = "combo__option";
      row.id = list.id + "-" + i;
      row.setAttribute("role", "option");
      row.setAttribute("aria-selected", "false");
      row.setAttribute("data-value", option.value);

      var name = document.createElement("span");
      name.className = "combo__value";
      name.textContent = option.value;
      row.appendChild(name);

      /* Preserve option labels containing connection and sensor-module details. */
      if (option.label && option.label !== option.value) {
        var note = document.createElement("span");
        note.className = "combo__note";
        note.textContent = option.label;
        row.appendChild(note);
      }
      list.appendChild(row);
    }

    combo.appendChild(list);
    combo.classList.add(OPEN);
    var toggle = combo.querySelector(".combo__toggle");
    if (toggle) toggle.setAttribute("aria-expanded", "true");
    input.setAttribute("aria-controls", list.id);
    open = combo;

    // Start at the current value so opening and pressing Enter does not change
    // selection.

    var rows = list.querySelectorAll(".combo__option");
    for (var j = 0; j < rows.length; j++) {
      if (rows[j].getAttribute("data-value") === input.value) {
        highlight(combo, j);
        break;
      }
    }
  }

  document.addEventListener("click", function (event) {
    var toggle = event.target.closest(".combo__toggle");
    if (toggle) {
      event.preventDefault();
      var combo = toggle.closest(".combo");
      if (combo === open) close();
      else openList(combo);
      return;
    }

    var row = event.target.closest(".combo__option");
    if (row) {
      choose(row.closest(".combo"), row.getAttribute("data-value"));
      return;
    }


    var box = event.target.closest("." + SEARCH + " .input");
    if (box) {
      if (box.closest(".combo") !== open) openList(box.closest(".combo"));
      return;
    }


    if (!event.target.closest(".combo--open")) close();
  });

  /* Open filtered search results only for real keystrokes, not the input event
     emitted by choosing an option. */
  document.addEventListener("input", function (event) {
    if (!event.isTrusted || !event.target.classList.contains("input")) return;
    var combo = event.target.closest("." + SEARCH);
    if (combo) openList(combo, event.target.value);
  });

  /* Close on blur, but allow a row press to complete before input focus is lost. */
  document.addEventListener("mousedown", function (event) {
    if (event.target.closest("." + SEARCH + " .combo__list")) event.preventDefault();
  });
  document.addEventListener("focusout", function (event) {
    if (open && open.classList.contains(SEARCH) && !open.contains(event.relatedTarget)) {
      close();
    }
  });

  document.addEventListener("keydown", function (event) {
    var combo = event.target.closest(".combo");
    if (!combo) return;

    if (event.key === "Escape" && open) {
      event.preventDefault();
      close();
      return;
    }
    if (event.key === "ArrowDown" && combo !== open) {
      // Use the input's keyboard route rather than adding a second tab stop for each
      // dropdown button.


      event.preventDefault();
      openList(combo);
      return;
    }
    if (combo !== open) return;

    if (event.key === "ArrowDown") {
      event.preventDefault();
      highlight(combo, selectedIndex(combo) + 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      highlight(combo, selectedIndex(combo) - 1);
    } else if (event.key === "Enter") {
      var at = selectedIndex(combo);
      var rows = combo.querySelectorAll(".combo__option");
      if (at >= 0 && rows[at]) {
        // Intercept Enter only while choosing a highlighted row; otherwise preserve
        // form submission.

        event.preventDefault();
        choose(combo, rows[at].getAttribute("data-value"));
      }
    }
  });

  /* Close lists before their anchor is replaced by HTMX. Keep the Task list open
     when only its derived status line changes. */
  document.addEventListener("htmx:beforeSwap", function (event) {
    var target = event.detail && event.detail.target;
    if (open && open.classList.contains(SEARCH) && target && !target.contains(open)) {
      return;
    }
    close();
  });

  /* Reveal dropdown controls after every swap; templates keep them hidden when
     JavaScript is unavailable. */
  function reveal(root) {
    var buttons = (root || document).querySelectorAll(".combo__toggle[hidden]");
    for (var i = 0; i < buttons.length; i++) buttons[i].hidden = false;
  }

  /* Move the native list binding so search inputs cannot open two popups while
     options remain discoverable. */
  function takeOverSearch(root) {
    var boxes = (root || document).querySelectorAll("." + SEARCH + " .input[list]");
    for (var i = 0; i < boxes.length; i++) {
      boxes[i].setAttribute("data-list", boxes[i].getAttribute("list"));
      boxes[i].removeAttribute("list");
    }
  }
  reveal();
  takeOverSearch();
  document.addEventListener("htmx:afterSwap", function (event) {
    reveal(event.target);
    takeOverSearch(event.target);
  });
})();
