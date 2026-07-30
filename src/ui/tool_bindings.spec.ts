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

import { describe, expect, it } from "vitest";
import type { ToolActivation } from "#src/ui/tool.js";
import { ToolBindingManager } from "#src/ui/tool_bindings.js";
import type { EventActionMap } from "#src/util/event_action_map.js";

function makeFakeActivation() {
  const installedMaps: EventActionMap[] = [];
  const activation = {
    inputEventMapBinder: (map: EventActionMap) => {
      installedMaps.push(map);
    },
  } as unknown as ToolActivation;
  return { activation, installedMaps };
}

function chips(indicator: HTMLElement) {
  return Array.from(
    indicator.querySelectorAll(".neuroglancer-tool-bind-chip"),
    (chip) => ({
      key: chip.querySelector(".neuroglancer-tool-bind-chip-key")?.textContent,
      label: chip.querySelector(".neuroglancer-tool-bind-chip-label")
        ?.textContent,
      active: chip.classList.contains("neuroglancer-tool-bind-chip-active"),
    }),
  );
}

describe("ToolBindingManager", () => {
  it("captures/indicates only enabled groups and re-installs when the key set changes", () => {
    const { activation, installedMaps } = makeFakeActivation();
    const indicator = document.createElement("div");
    const manager = new ToolBindingManager(activation, { indicator });

    let g2enabled = true;
    let twoActive = false;
    manager.addGroup({
      id: "g1",
      bindings: [
        { eventKey: "digit1", action: "a1", handler: () => {}, label: "One" },
      ],
    });
    manager.addGroup({
      id: "g2",
      enabled: () => g2enabled,
      bindings: [
        {
          eventKey: "digit2",
          action: "a2",
          handler: () => {},
          label: "Two",
          active: () => twoActive,
        },
        // A silent (unlabeled) bind: captured but never shown as a chip.
        { eventKey: "enter", action: "a2-commit", handler: () => {} },
      ],
    });

    // Both groups enabled: both labeled binds shown; the silent bind is captured
    // (in the installed map) but not shown.
    manager.update();
    expect(chips(indicator).map((c) => c.label)).toEqual(["One", "Two"]);
    expect(installedMaps).toHaveLength(1);
    expect(installedMaps[0].describe()).toContain("a2-commit");

    // Only the `active` predicate changed (same key set): re-render, no re-install.
    twoActive = true;
    manager.update();
    expect(chips(indicator).find((c) => c.label === "Two")?.active).toBe(true);
    expect(installedMaps).toHaveLength(1);

    // Disable g2 (sub-mode change): its keys leave the captured set and its chip
    // disappears; g1 remains.
    g2enabled = false;
    manager.update();
    expect(chips(indicator).map((c) => c.label)).toEqual(["One"]);
    expect(installedMaps).toHaveLength(2);
    expect(installedMaps[1].describe()).not.toContain("a2");

    manager.dispose();
  });
});
