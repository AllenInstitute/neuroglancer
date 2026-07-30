/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Shared infrastructure for tools that capture their own keybinds while active and show a
 * flexible, conditional "binds indicator" — instead of each tool re-implementing key capture +
 * status rendering on top of `ToolActivation.bindInputEventMap`/`bindAction`.
 *
 * The pieces:
 *  - `ToolBinding` / `ToolBindingGroup`: a declarative model of the keys a tool captures. A bind
 *    with a `label` is shown as an indicator chip; without one it is bound silently (curated vs
 *    hidden). A group has an optional `enabled` predicate for sub-modes.
 *  - `ToolBindingManager`: installs the active groups' keys at high priority (so they override
 *    defaults only while active and every unbound key falls through), and — when given an
 *    `indicator` element — renders the currently-relevant binds as chips. `update()` re-evaluates
 *    which groups are enabled, re-installs the key set only if it changed, and re-renders the
 *    (conditional) indicator, so a tool can swap its active binds by sub-mode / state.
 *  - `renderBindChips`: the shared chip renderer (also usable directly).
 *  - `ToolValueEntryBuffer`: a keyed value accumulator (type digits, Enter to commit, Esc to
 *    exit) rendered as a dark box that turns red when out of range — for numeric/enum entry.
 */

import "#src/ui/tool.css";

import type { ToolActivation } from "#src/ui/tool.js";
import { RefCounted } from "#src/util/disposable.js";
import { removeChildren } from "#src/util/dom.js";
import type { ActionEvent } from "#src/util/event_action_map.js";
import {
  EventActionMap,
  registerActionListener,
} from "#src/util/event_action_map.js";

type MaybeThunk<T> = T | (() => T);

function resolve<T>(value: MaybeThunk<T>): T {
  return typeof value === "function" ? (value as () => T)() : value;
}

export interface ToolBinding {
  // Event identifier, e.g. "digit1", "keyw", "at:control+mousedown0".
  eventKey: string;
  // Unique action id for this activation.
  action: string;
  handler: (event: ActionEvent<Event>) => void;
  // If provided, the bind is shown as an indicator chip; otherwise it is bound silently.
  label?: MaybeThunk<string>;
  // Chip key text; defaults to a friendly rendering of `eventKey`.
  keyLabel?: MaybeThunk<string>;
  // Chip highlighted (e.g. the current value).
  active?: () => boolean;
  // Chip shown at all; defaults to true.
  visible?: () => boolean;
}

export interface ToolBindingGroup {
  id: string;
  // Whether this group's binds are currently active; defaults to always enabled.
  enabled?: () => boolean;
  bindings: ToolBinding[];
}

// Friendly key text for a chip when no explicit `keyLabel` is given: "digit1" -> "1",
// "keyw" -> "W". Falls back to the raw identifier for anything else (e.g. mouse binds).
function friendlyKeyLabel(eventKey: string): string {
  const match = /^(?:digit|key)([a-z0-9])$/.exec(eventKey);
  return match !== null ? match[1].toUpperCase() : eventKey;
}

export interface BindChip {
  key: string;
  label: string;
  active?: boolean;
}

// Renders key/label chips into `container` (clears it first).
export function renderBindChips(container: HTMLElement, chips: BindChip[]) {
  removeChildren(container);
  for (const chip of chips) {
    const el = document.createElement("div");
    el.classList.add("neuroglancer-tool-bind-chip");
    if (chip.active) {
      el.classList.add("neuroglancer-tool-bind-chip-active");
    }
    const keyEl = document.createElement("span");
    keyEl.classList.add("neuroglancer-tool-bind-chip-key");
    keyEl.textContent = chip.key;
    const labelEl = document.createElement("span");
    labelEl.classList.add("neuroglancer-tool-bind-chip-label");
    labelEl.textContent = chip.label;
    el.appendChild(keyEl);
    el.appendChild(labelEl);
    container.appendChild(el);
  }
}

export class ToolBindingManager extends RefCounted {
  private groups: ToolBindingGroup[] = [];
  private installation: RefCounted | undefined;
  private installedSignature: string | undefined;
  private readonly indicator: HTMLElement | undefined;

  constructor(
    private readonly activation: ToolActivation,
    options: { indicator?: HTMLElement } = {},
  ) {
    super();
    this.indicator = options.indicator;
  }

  addGroup(group: ToolBindingGroup) {
    this.groups.push(group);
    return this;
  }

  setGroups(groups: ToolBindingGroup[]) {
    this.groups = groups;
    return this;
  }

  private activeBindings(): ToolBinding[] {
    const result: ToolBinding[] = [];
    for (const group of this.groups) {
      if (group.enabled !== undefined && !group.enabled()) continue;
      result.push(...group.bindings);
    }
    return result;
  }

  // Re-evaluate which groups are enabled; (re)install the active key set if it changed; and
  // re-render the conditional binds indicator. Call after any state change that affects a
  // group's `enabled` or a bind's `active`/`visible`/`label`.
  update() {
    const bindings = this.activeBindings();
    const signature = bindings
      .map((b) => `${b.eventKey}=${b.action}`)
      .join("\n");
    if (signature !== this.installedSignature) {
      this.installedSignature = signature;
      this.installation?.dispose();
      const installation = (this.installation = new RefCounted());
      const map: { [eventKey: string]: string } = {};
      for (const b of bindings) {
        map[b.eventKey] = b.action;
      }
      // Install at POSITIVE_INFINITY (via the activation's binder) so these keys override
      // defaults only while active; unbound keys fall through to normal keybinds.
      this.activation.inputEventMapBinder(
        EventActionMap.fromObject(map),
        installation,
      );
      for (const b of bindings) {
        installation.registerDisposer(
          registerActionListener(window, b.action, b.handler),
        );
      }
    }
    this.renderIndicator(bindings);
  }

  private renderIndicator(bindings: ToolBinding[]) {
    const { indicator } = this;
    if (indicator === undefined) return;
    const chips: BindChip[] = [];
    for (const b of bindings) {
      if (b.label === undefined) continue;
      if (b.visible !== undefined && !b.visible()) continue;
      chips.push({
        key:
          b.keyLabel !== undefined
            ? resolve(b.keyLabel)
            : friendlyKeyLabel(b.eventKey),
        label: resolve(b.label),
        active: b.active?.() ?? false,
      });
    }
    renderBindChips(indicator, chips);
  }

  disposed() {
    this.installation?.dispose();
    this.installation = undefined;
    super.disposed();
  }
}

export interface ToolValueEntryOptions {
  // All read live because they depend on tool/property state.
  isFloat: () => boolean;
  allowNegative: () => boolean;
  // Inclusive [min, max] range.
  range: () => [number, number];
  // The value shown when nothing is being typed, or undefined.
  currentValue: () => number | undefined;
  // Commit a validated, in-range value.
  commit: (value: number) => void;
}

export interface ToolValueEntryBindOptions {
  // Re-render the value display (called after any buffer change).
  requestRefresh: () => void;
  // Exit the mode (Escape).
  cancel: () => void;
  // Keeps action ids unique when multiple buffers coexist.
  actionPrefix?: string;
}

// A keyed numeric value accumulator: only number-related keys are routed to it (leaving every
// other keybind live). Type digits (and "." for floats, "-" for signed), Enter to commit, Esc
// to exit; the buffer renders as a dark box that turns red when the value is out of range and
// refuses to commit an invalid value.
export class ToolValueEntryBuffer {
  private buffer = "";

  constructor(private readonly options: ToolValueEntryOptions) {}

  reset() {
    this.buffer = "";
  }

  private parse(): number | undefined {
    if (this.buffer === "" || this.buffer === "-" || this.buffer === ".") {
      return undefined;
    }
    const value = this.options.isFloat()
      ? Number.parseFloat(this.buffer)
      : Number.parseInt(this.buffer, 10);
    return Number.isFinite(value) ? value : undefined;
  }

  private get outOfRange(): boolean {
    const value = this.parse();
    if (value === undefined) return false;
    const [min, max] = this.options.range();
    return value < min || value > max;
  }

  // The (silent, unlabeled) key bindings for this buffer, to add to a `ToolBindingManager`
  // group.
  bindings(bindOptions: ToolValueEntryBindOptions): ToolBinding[] {
    const {
      requestRefresh,
      cancel,
      actionPrefix = "tool-value-entry",
    } = bindOptions;
    const append = (ch: string) => (event: ActionEvent<Event>) => {
      event.stopPropagation();
      if (ch === "-" && this.buffer.length > 0) return; // leading only
      if (ch === "." && this.buffer.includes(".")) return; // one decimal point
      this.buffer += ch;
      requestRefresh();
    };
    const list: ToolBinding[] = [];
    for (let digit = 0; digit <= 9; ++digit) {
      list.push({
        eventKey: `digit${digit}`,
        action: `${actionPrefix}-${digit}`,
        handler: append(String(digit)),
      });
    }
    if (this.options.isFloat()) {
      list.push({
        eventKey: "period",
        action: `${actionPrefix}-decimal`,
        handler: append("."),
      });
    }
    if (this.options.allowNegative()) {
      list.push({
        eventKey: "minus",
        action: `${actionPrefix}-minus`,
        handler: append("-"),
      });
    }
    list.push({
      eventKey: "backspace",
      action: `${actionPrefix}-backspace`,
      handler: (event) => {
        event.stopPropagation();
        this.buffer = this.buffer.slice(0, -1);
        requestRefresh();
      },
    });
    list.push({
      eventKey: "enter",
      action: `${actionPrefix}-commit`,
      handler: (event) => {
        event.stopPropagation();
        const value = this.parse();
        // Refuse an empty/incomplete or out-of-range value; the buffer stays (shown red).
        if (value === undefined || this.outOfRange) return;
        this.options.commit(value);
        this.buffer = "";
        requestRefresh();
      },
    });
    list.push({
      eventKey: "escape",
      action: `${actionPrefix}-exit`,
      handler: (event) => {
        event.stopPropagation();
        cancel();
      },
    });
    return list;
  }

  // Appends the value box to `container` (caller clears it first).
  render(container: HTMLElement) {
    const box = document.createElement("div");
    box.classList.add("neuroglancer-tool-entry-box");
    box.textContent =
      this.buffer !== ""
        ? this.buffer
        : `${this.options.currentValue() ?? "—"}`;
    if (this.outOfRange) {
      box.classList.add("neuroglancer-tool-entry-box-invalid");
    }
    container.appendChild(box);
  }
}
