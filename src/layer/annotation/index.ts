/**
 * @license
 * Copyright 2016 Google Inc.
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

import "#src/layer/annotation/style.css";

import type { AnnotationDisplayState } from "#src/annotation/annotation_layer_state.js";
import { AnnotationLayerState } from "#src/annotation/annotation_layer_state.js";
import { MultiscaleAnnotationSource } from "#src/annotation/frontend_source.js";
import type {
  AnnotationNumericPropertySpec,
  AnnotationPropertySpec,
} from "#src/annotation/index.js";
import {
  annotationPropertySpecsToJson,
  AnnotationType,
  isAnnotationNumericPropertySpec,
  LocalAnnotationSource,
  parseAnnotationPropertySpecs,
  propertyTypeDataType,
} from "#src/annotation/index.js";
import type { CoordinateTransformSpecification } from "#src/coordinate_transform.js";
import { makeCoordinateSpace } from "#src/coordinate_transform.js";
import type { DataSourceSpecification } from "#src/datasource/index.js";
import { localAnnotationsUrl, LocalDataSource } from "#src/datasource/local.js";
import { buildShaderPropertyList } from "#src/layer/annotation/shader_ui_property_list.js";
import type { LayerManager, ManagedUserLayer } from "#src/layer/index.js";
import {
  LayerReference,
  registerLayerType,
  registerLayerTypeDetector,
  UserLayer,
} from "#src/layer/index.js";
import type { LoadedDataSubsource } from "#src/layer/layer_data_source.js";
import { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { Overlay } from "#src/overlay.js";
import { getWatchableRenderLayerTransform } from "#src/render_coordinate_transform.js";
import { RenderLayerRole } from "#src/renderlayer.js";
import type { SegmentationDisplayState } from "#src/segmentation_display_state/frontend.js";
import {
  ElementVisibilityFromTrackableBoolean,
  TrackableBoolean,
  TrackableBooleanCheckbox,
} from "#src/trackable_boolean.js";
import {
  makeCachedLazyDerivedWatchableValue,
  WatchableValue,
  observeWatchable,
} from "#src/trackable_value.js";
import {
  annotateEnumPropertyToolJson,
  annotateNumberPropertyToolJson,
  toggleBoolPropertyToolJson,
} from "#src/ui/annotation_properties.js";
import { AnnotationSchemaTab } from "#src/ui/annotation_schema_tab.js";
import type {
  AnnotationLayerView,
  MergedAnnotationStates,
} from "#src/ui/annotations.js";
import {
  SELECT_NEXT_ANNOTATION_TOOL_ID,
  SELECT_PREVIOUS_ANNOTATION_TOOL_ID,
  UserLayerWithAnnotationsMixin,
} from "#src/ui/annotations.js";
import type { ToolActivation } from "#src/ui/tool.js";
import {
  getToolBoundKey,
  LayerTool,
  makeToolActivationStatusMessageWithHeader,
  registerTool,
  unregisterTool,
} from "#src/ui/tool.js";
import type { ToolBinding } from "#src/ui/tool_bindings.js";
import {
  renderBindChips,
  ToolBindingManager,
  ToolValueEntryBuffer,
} from "#src/ui/tool_bindings.js";
import { animationFrameDebounce } from "#src/util/animation_frame_debounce.js";
import { DataType } from "#src/util/data_type.js";
import type { Borrowed, Owned } from "#src/util/disposable.js";
import { RefCounted } from "#src/util/disposable.js";
import { removeChildren, updateChildren } from "#src/util/dom.js";
import {
  parseArray,
  parseFixedLengthArray,
  stableStringify,
  verify3dVec,
  verifyFinitePositiveFloat,
  verifyObject,
  verifyOptionalObjectProperty,
  verifyString,
  verifyStringArray,
} from "#src/util/json.js";
import { defaultDataTypeRange } from "#src/util/lerp.js";
import { NullarySignal } from "#src/util/signal.js";
import { DependentViewWidget } from "#src/widget/dependent_view_widget.js";
import {
  addLayerControlToOptionsTab,
  type LayerControlDefinition,
  registerLayerControl,
} from "#src/widget/layer_control.js";
import { colorLayerControl } from "#src/widget/layer_control_color.js";
import { LayerReferenceWidget } from "#src/widget/layer_reference.js";
import { RenderScaleWidget } from "#src/widget/render_scale_widget.js";
import {
  makeShaderCodeWidgetTopRow,
  ShaderCodeWidget,
} from "#src/widget/shader_code_widget.js";
import {
  registerLayerShaderControlsTool,
  ShaderControls,
} from "#src/widget/shader_controls.js";
import { Tab } from "#src/widget/tab_view.js";

const POINTS_JSON_KEY = "points";
const ANNOTATIONS_JSON_KEY = "annotations";
const ANNOTATION_PROPERTIES_JSON_KEY = "annotationProperties";
const ANNOTATION_RELATIONSHIPS_JSON_KEY = "annotationRelationships";
const CROSS_SECTION_RENDER_SCALE_JSON_KEY = "crossSectionAnnotationSpacing";
const PROJECTION_RENDER_SCALE_JSON_KEY = "projectionAnnotationSpacing";
const SHADER_JSON_KEY = "shader";
const SHADER_CONTROLS_JSON_KEY = "shaderControls";
const ANNOTATION_COLOR_JSON_KEY = "annotationColor";

function addPointAnnotations(annotations: LocalAnnotationSource, obj: any) {
  if (obj === undefined) {
    return;
  }
  parseArray(obj, (x, i) => {
    annotations.add({
      type: AnnotationType.POINT,
      id: "" + i,
      point: verify3dVec(x),
      properties: [],
    });
  });
}

function isValidLinkedSegmentationLayer(layer: ManagedUserLayer) {
  const userLayer = layer.layer;
  if (userLayer === null) {
    return true;
  }
  if (userLayer instanceof SegmentationUserLayer) {
    return true;
  }
  return false;
}

function getSegmentationDisplayState(
  layer: ManagedUserLayer | undefined,
): SegmentationDisplayState | null {
  if (layer === undefined) {
    return null;
  }
  const userLayer = layer.layer;
  if (userLayer === null) {
    return null;
  }
  if (!(userLayer instanceof SegmentationUserLayer)) {
    return null;
  }
  return userLayer.displayState;
}

interface LinkedSegmentationLayer {
  layerRef: Owned<LayerReference>;
  showMatches: TrackableBoolean;
  seenGeneration: number;
}

const LINKED_SEGMENTATION_LAYER_JSON_KEY = "linkedSegmentationLayer";
const FILTER_BY_SEGMENTATION_JSON_KEY = "filterBySegmentation";
const IGNORE_NULL_SEGMENT_FILTER_JSON_KEY = "ignoreNullSegmentFilter";
const CODE_VISIBLE_KEY = "codeVisible";
const HIDE_INACTIVE_SHADER_CONTROLS_JSON_KEY = "hideInactiveShaderControls";

class LinkedSegmentationLayers extends RefCounted {
  changed = new NullarySignal();
  private curGeneration = -1;
  private wasLoading: boolean | undefined = undefined;
  constructor(
    public layerManager: Borrowed<LayerManager>,
    public annotationStates: Borrowed<MergedAnnotationStates>,
    public annotationDisplayState: Borrowed<AnnotationDisplayState>,
  ) {
    super();
    this.registerDisposer(annotationStates.changed.add(() => this.update()));
    this.registerDisposer(
      annotationStates.isLoadingChanged.add(() => this.update()),
    );
    this.update();
  }

  private update() {
    const generation = this.annotationStates.changed.count;
    const isLoading = this.annotationStates.isLoading;
    if (this.curGeneration === generation && isLoading === this.wasLoading)
      return;
    this.wasLoading = isLoading;
    this.curGeneration = generation;
    const { map } = this;
    let changed = false;
    for (const relationship of this.annotationStates.relationships) {
      let state = map.get(relationship);
      if (state === undefined) {
        state = this.addRelationship(relationship);
        changed = true;
      }
      state.seenGeneration = generation;
    }
    if (!isLoading) {
      const { relationshipStates } = this.annotationDisplayState;
      for (const [relationship, state] of map) {
        if (state.seenGeneration !== generation) {
          map.delete(relationship);
          relationshipStates.delete(relationship);
          changed = true;
        }
      }
    }
    if (changed) {
      this.changed.dispatch();
    }
  }

  private addRelationship(relationship: string): LinkedSegmentationLayer {
    const relationshipState =
      this.annotationDisplayState.relationshipStates.get(relationship);
    const layerRef = new LayerReference(
      this.layerManager.addRef(),
      isValidLinkedSegmentationLayer,
    );
    layerRef.registerDisposer(
      layerRef.changed.add(() => {
        relationshipState.segmentationState.value =
          layerRef.layerName === undefined
            ? undefined
            : getSegmentationDisplayState(layerRef.layer);
      }),
    );
    const { showMatches } = relationshipState;
    const state = {
      layerRef,
      showMatches,
      seenGeneration: -1,
    };
    layerRef.changed.add(this.changed.dispatch);
    showMatches.changed.add(this.changed.dispatch);
    this.map.set(relationship, state);
    return state;
  }

  get(relationship: string): LinkedSegmentationLayer {
    this.update();
    return this.map.get(relationship)!;
  }

  private unbind(state: LinkedSegmentationLayer) {
    state.layerRef.changed.remove(this.changed.dispatch);
    state.showMatches.changed.remove(this.changed.dispatch);
  }

  reset() {
    for (const state of this.map.values()) {
      state.showMatches.reset();
    }
  }

  toJSON() {
    const { map } = this;
    if (map.size === 0) return {};
    let linkedJson: { [relationship: string]: string } | undefined = undefined;
    const filterBySegmentation = [];
    for (const [name, state] of map) {
      if (state.showMatches.value) {
        filterBySegmentation.push(name);
      }
      const { layerName } = state.layerRef;
      if (layerName !== undefined) {
        (linkedJson = linkedJson || {})[name] = layerName;
      }
    }
    filterBySegmentation.sort();
    return {
      [LINKED_SEGMENTATION_LAYER_JSON_KEY]: linkedJson,
      [FILTER_BY_SEGMENTATION_JSON_KEY]:
        filterBySegmentation.length === 0 ? undefined : filterBySegmentation,
    };
  }
  restoreState(json: any) {
    const { isLoading } = this.annotationStates;
    verifyOptionalObjectProperty(
      json,
      LINKED_SEGMENTATION_LAYER_JSON_KEY,
      (linkedJson) => {
        if (typeof linkedJson === "string") {
          linkedJson = { segments: linkedJson };
        }
        verifyObject(linkedJson);
        for (const key of Object.keys(linkedJson)) {
          const value = verifyString(linkedJson[key]);
          let state = this.map.get(key);
          if (state === undefined) {
            if (!isLoading) continue;
            state = this.addRelationship(key);
          }
          state.layerRef.layerName = value;
        }
        for (const [relationship, state] of this.map) {
          if (!Object.prototype.hasOwnProperty.call(linkedJson, relationship)) {
            state.layerRef.layerName = undefined;
          }
        }
      },
    );
    verifyOptionalObjectProperty(
      json,
      FILTER_BY_SEGMENTATION_JSON_KEY,
      (filterJson) => {
        if (typeof filterJson === "boolean") {
          filterJson = filterJson === true ? ["segments"] : [];
        }
        for (const key of verifyStringArray(filterJson)) {
          let state = this.map.get(key);
          if (state === undefined) {
            if (!isLoading) continue;
            state = this.addRelationship(key);
          }
          state.showMatches.value = true;
        }
      },
    );
  }

  disposed() {
    const { map } = this;
    for (const state of map.values()) {
      this.unbind(state);
    }
    map.clear();
    super.disposed();
  }
  private map = new Map<string, LinkedSegmentationLayer>();
}

class LinkedSegmentationLayerWidget extends RefCounted {
  element = document.createElement("label");
  seenGeneration = -1;
  constructor(
    public relationship: string,
    public state: LinkedSegmentationLayer,
  ) {
    super();
    const { element } = this;
    const checkboxWidget = this.registerDisposer(
      new TrackableBooleanCheckbox(state.showMatches),
    );
    const layerWidget = new LayerReferenceWidget(state.layerRef);
    element.appendChild(checkboxWidget.element);
    element.appendChild(document.createTextNode(relationship));
    element.appendChild(layerWidget.element);
  }
}

class LinkedSegmentationLayersWidget extends RefCounted {
  widgets = new Map<string, LinkedSegmentationLayerWidget>();
  element = document.createElement("div");
  constructor(public linkedSegmentationLayers: LinkedSegmentationLayers) {
    super();
    this.element.style.display = "contents";
    const debouncedUpdateView = this.registerCancellable(
      animationFrameDebounce(() => this.updateView()),
    );
    this.registerDisposer(
      this.linkedSegmentationLayers.annotationStates.changed.add(
        debouncedUpdateView,
      ),
    );
    this.updateView();
  }

  private updateView() {
    const { linkedSegmentationLayers } = this;
    const { annotationStates } = linkedSegmentationLayers;
    const generation = annotationStates.changed.count;
    const { widgets } = this;
    function* getChildren(this: LinkedSegmentationLayersWidget) {
      for (const relationship of annotationStates.relationships) {
        let widget = widgets.get(relationship);
        if (widget === undefined) {
          widget = new LinkedSegmentationLayerWidget(
            relationship,
            linkedSegmentationLayers.get(relationship),
          );
        }
        widget.seenGeneration = generation;
        yield widget.element;
      }
    }
    for (const [relationship, widget] of widgets) {
      if (widget.seenGeneration !== generation) {
        widget.dispose();
        widgets.delete(relationship);
      }
    }
    updateChildren(this.element, getChildren.call(this));
  }

  disposed() {
    super.disposed();
    for (const widget of this.widgets.values()) {
      widget.dispose();
    }
  }
}

// Identifies a keybindable property-entry tool so it can be (re)constructed,
// one per property type.
type PropertyToolDescriptor =
  | { kind: "enum"; propertyIdentifier: string }
  | { kind: "bool"; propertyIdentifier: string }
  | { kind: "number"; propertyIdentifier: string };

const propertyToolRegistrationCounts = new Map<string, number>();

function retainPropertyTool(
  toolId: string,
  descriptor: PropertyToolDescriptor,
) {
  const registrationCount = propertyToolRegistrationCounts.get(toolId) ?? 0;
  if (registrationCount === 0) {
    registerTool(AnnotationUserLayer, toolId, (layer) => {
      switch (descriptor.kind) {
        case "enum":
          return new EnumPropertyEntryTool(
            descriptor.propertyIdentifier,
            layer,
          );
        case "bool":
          return new ToggleBoolPropertyTool(
            descriptor.propertyIdentifier,
            layer,
          );
        case "number":
          return new NumberPropertyEntryTool(
            descriptor.propertyIdentifier,
            layer,
          );
      }
    });
  }
  propertyToolRegistrationCounts.set(toolId, registrationCount + 1);
}

function releasePropertyTool(toolId: string) {
  const registrationCount = propertyToolRegistrationCounts.get(toolId);
  if (registrationCount === undefined) return;
  if (registrationCount === 1) {
    propertyToolRegistrationCounts.delete(toolId);
    unregisterTool(AnnotationUserLayer, toolId);
  } else {
    propertyToolRegistrationCounts.set(toolId, registrationCount - 1);
  }
}

// Sets a single property of the currently-selected annotation, computing the
// new value from the existing one. Shared by the property-entry tools below.
function updateSelectedAnnotationProperty(
  layer: AnnotationUserLayer,
  propertyIdentifier: string,
  computeValue: (currentValue: number) => number,
) {
  const context = layer.getSelectedAnnotationContext();
  if (context === undefined) return;
  const { annotationLayerState, annotationId } = context;
  const { source } = annotationLayerState;
  const propertyIndex = source.properties.value.findIndex(
    (x) => x.identifier === propertyIdentifier,
  );
  if (propertyIndex === -1) return;
  const reference = source.getReference(annotationId);
  try {
    const annotation = reference.value;
    if (annotation != null) {
      const properties = annotation.properties.slice();
      properties[propertyIndex] = computeValue(properties[propertyIndex]);
      source.update(reference, { ...annotation, properties });
      source.commit(reference);
    }
  } finally {
    reference.dispose();
  }
}

// Returns the currently-selected annotation's value for `propertyIdentifier`,
// or undefined if there is no selection / property.
function getSelectedAnnotationPropertyValue(
  layer: AnnotationUserLayer,
  propertyIdentifier: string,
): number | undefined {
  const context = layer.getSelectedAnnotationContext();
  if (context === undefined) return undefined;
  const { source } = context.annotationLayerState;
  const propertyIndex = source.properties.value.findIndex(
    (x) => x.identifier === propertyIdentifier,
  );
  if (propertyIndex === -1) return undefined;
  const reference = source.getReference(context.annotationId);
  try {
    const value = reference.value?.properties[propertyIndex];
    return typeof value === "number" ? value : undefined;
  } finally {
    reference.dispose();
  }
}

// Maps an option index to the number key that triggers it: 1-9 for the first
// nine options, 0 for a tenth. Further options have no key.
function keyForEnumOptionIndex(index: number): string | undefined {
  if (index < 9) return String(index + 1);
  if (index === 9) return "0";
  return undefined;
}

// Base class for the keybindable, toggle-style "data-entry mode" tools. While
// active, the tool stays active across annotation navigation and shows a
// bottom-of-screen status notification. It binds the unshifted keys of the
// layer's previous/next annotation tools to move within the mode, and delegates
// the value-entry keys (number keys, typing, ...) and the rendering of the
// current value to the concrete subclass.
abstract class AnnotationPropertyEntryTool extends LayerTool<AnnotationUserLayer> {
  constructor(
    public propertyIdentifier: string,
    layer: AnnotationUserLayer,
  ) {
    super(layer, /*toggle=*/ true);
  }

  protected get property(): AnnotationPropertySpec | undefined {
    return this.layer.localAnnotations?.properties.value?.find(
      (x) => x.identifier === this.propertyIdentifier,
    );
  }

  protected get currentValue(): number | undefined {
    return getSelectedAnnotationPropertyValue(
      this.layer,
      this.propertyIdentifier,
    );
  }

  // Verb shown in the status header, e.g. "Set".
  protected abstract readonly modeVerb: string;

  // Declare the value-entry key bindings for this property kind by adding groups
  // to the shared `ToolBindingManager`. `requestRefresh` re-renders the display;
  // `activation` lets a binding end the mode (Escape). No-op by default.
  protected configureBindings(
    _manager: ToolBindingManager,
    _activation: ToolActivation<this>,
    _requestRefresh: () => void,
  ): void {}

  // Render the value area (`valueContainer`, e.g. option chips) and, for the
  // buffered numeric modes, the inline entry box into `entrySlot` (which sits on
  // the nav line before the prev/next hint). No-op by default.
  protected renderValue(
    _valueContainer: HTMLElement,
    _entrySlot: HTMLElement,
  ): void {}

  // Called when the selected annotation changes, so transient entry state
  // (e.g. a number being typed) can be reset. No-op by default.
  protected onAnnotationChanged(): void {}

  // Extra text appended to the navigation hint (e.g. commit/exit keys for the
  // buffered numeric modes). Empty by default.
  protected navHintSuffix(): string {
    return "";
  }

  // The unshifted key-event identifiers bound to this layer's previous/next
  // annotation tools, and a human-readable hint describing them.
  protected navKeyInfo(): {
    prevKey: string | undefined;
    nextKey: string | undefined;
    hint: string;
  } {
    const prevKey = getToolBoundKey(
      this.layer.toolBinder,
      SELECT_PREVIOUS_ANNOTATION_TOOL_ID,
    );
    const nextKey = getToolBoundKey(
      this.layer.toolBinder,
      SELECT_NEXT_ANNOTATION_TOOL_ID,
    );
    const labels: string[] = [];
    if (prevKey !== undefined)
      labels.push(`${prevKey.slice(3).toUpperCase()} prev`);
    if (nextKey !== undefined)
      labels.push(`${nextKey.slice(3).toUpperCase()} next`);
    const hint =
      labels.length > 0
        ? `${labels.join(" · ")} annotation`
        : "bind the prev/next annotation tools to navigate here";
    return { prevKey, nextKey, hint };
  }

  activate(activation: ToolActivation<this>) {
    const { layer, propertyIdentifier } = this;
    // The tool renders its own conditional indicator, so suppress the default
    // binds line.
    const { body, header } = makeToolActivationStatusMessageWithHeader(
      activation,
      { showBindings: false },
    );
    const valueContainer = document.createElement("div");
    valueContainer.classList.add("neuroglancer-annotation-entry-tool-values");
    body.appendChild(valueContainer);
    // Second line: an optional inline entry box (numeric modes) followed by the
    // navigation / commit hint.
    const navLine = document.createElement("div");
    navLine.classList.add("neuroglancer-annotation-entry-tool-nav-line");
    const entrySlot = document.createElement("span");
    navLine.appendChild(entrySlot);
    const navText = document.createElement("span");
    navText.classList.add("neuroglancer-annotation-entry-tool-nav");
    navLine.appendChild(navText);
    body.appendChild(navLine);

    const { prevKey, nextKey, hint } = this.navKeyInfo();
    const refresh = () => {
      const property = this.property;
      header.textContent =
        property === undefined
          ? `Property "${propertyIdentifier}" unavailable`
          : `${this.modeVerb} ${propertyIdentifier}`;
      this.renderValue(valueContainer, entrySlot);
      navText.textContent = hint + this.navHintSuffix();
    };
    const debouncedRefresh = activation.registerCancellable(
      animationFrameDebounce(refresh),
    );

    // Key capture via the shared binding manager: only these keys are
    // intercepted (overriding e.g. layer select/deselect while active); every
    // other keybind stays live. Value-entry keys come from the subclass; the
    // nav keys reuse the layer's prev/next tool bindings.
    const manager = activation.registerDisposer(
      new ToolBindingManager(activation),
    );
    this.configureBindings(manager, activation, debouncedRefresh);
    const navBindings: ToolBinding[] = [];
    if (prevKey !== undefined) {
      navBindings.push({
        eventKey: prevKey,
        action: "annotation-entry-prev",
        handler: (event) => {
          event.stopPropagation();
          layer.shiftSelectedIndexBy(-1);
        },
      });
    }
    if (nextKey !== undefined) {
      navBindings.push({
        eventKey: nextKey,
        action: "annotation-entry-next",
        handler: (event) => {
          event.stopPropagation();
          layer.shiftSelectedIndexBy(1);
        },
      });
    }
    manager.addGroup({ id: "nav", bindings: navBindings });
    manager.update();

    // Keep the display in sync and reset transient state as the selection
    // changes.
    activation.registerDisposer(
      layer.manager.root.selectionState.changed.add(() => {
        this.onAnnotationChanged();
        debouncedRefresh();
      }),
    );
    // Value bindings and numeric bounds are fixed when the mode is activated.
    // End the mode on schema changes so it cannot continue with stale input
    // behavior; activating it again rebuilds everything from the new schema.
    activation.registerDisposer(
      layer.localAnnotationProperties.changed.add(() => activation.cancel()),
    );
    this.onAnnotationChanged();
    refresh();
  }

  get description() {
    return `${this.modeVerb.toLowerCase()} ${this.propertyIdentifier}`;
  }
}

// Enum property: with up to ten options, number keys 1..9/0 set each option's
// value directly. With more options (too many for one key each), the user types
// the option's number and presses Enter instead.
class EnumPropertyEntryTool extends AnnotationPropertyEntryTool {
  protected readonly modeVerb = "Set";

  // For >10 options, the user types the option's 1-based number instead of a
  // single key. Values are 1..N integers.
  private readonly entry = new ToolValueEntryBuffer({
    isFloat: () => false,
    allowNegative: () => false,
    range: () => [1, this.numericProperty?.enumValues?.length ?? 0],
    currentValue: () => {
      const enumValues = this.numericProperty?.enumValues ?? [];
      const index =
        this.currentValue === undefined
          ? -1
          : enumValues.indexOf(this.currentValue);
      return index === -1 ? undefined : index + 1;
    },
    commit: (optionNumber) => {
      const value = this.numericProperty?.enumValues?.[optionNumber - 1];
      if (value === undefined) return;
      updateSelectedAnnotationProperty(
        this.layer,
        this.propertyIdentifier,
        () => value,
      );
    },
  });

  private get numericProperty(): AnnotationNumericPropertySpec | undefined {
    const property = this.property;
    return property !== undefined && isAnnotationNumericPropertySpec(property)
      ? property
      : undefined;
  }

  private get useNumberEntry(): boolean {
    return (this.numericProperty?.enumValues?.length ?? 0) > 10;
  }

  protected configureBindings(
    manager: ToolBindingManager,
    activation: ToolActivation<this>,
    requestRefresh: () => void,
  ) {
    if (this.useNumberEntry) {
      // Too many options for one key each: type the option's number + Enter.
      manager.addGroup({
        id: "enum-number",
        bindings: this.entry.bindings({
          requestRefresh,
          cancel: () => activation.cancel(),
          actionPrefix: "annotation-enum-number",
        }),
      });
      return;
    }
    // Direct key-per-option: number keys 1..9/0 set each option's value.
    const enumValues = this.numericProperty?.enumValues ?? [];
    const boundOptionCount = Math.min(enumValues.length, 10);
    const bindings: ToolBinding[] = [];
    for (let index = 0; index < boundOptionCount; ++index) {
      bindings.push({
        eventKey: `digit${keyForEnumOptionIndex(index)}`,
        action: `annotation-enum-option-${index}`,
        handler: (event) => {
          event.stopPropagation();
          const value = this.numericProperty?.enumValues?.[index];
          if (value === undefined) return;
          updateSelectedAnnotationProperty(
            this.layer,
            this.propertyIdentifier,
            () => value,
          );
          requestRefresh();
        },
      });
    }
    manager.addGroup({ id: "enum-keys", bindings });
  }

  protected renderValue(valueContainer: HTMLElement, entrySlot: HTMLElement) {
    removeChildren(entrySlot);
    const property = this.numericProperty;
    const enumValues = property?.enumValues ?? [];
    const enumLabels = property?.enumLabels ?? [];
    const currentValue = this.currentValue;
    const useNumberEntry = this.useNumberEntry;
    // The chip's leading number differs by mode: in key-per-option mode it is
    // the single key that selects it (1..9, then 0 for the 10th); in numeric
    // mode it is the 1-based number you type (1, 2, ..., 10, 11, ...).
    renderBindChips(
      valueContainer,
      enumValues.map((value, index) => ({
        key: useNumberEntry
          ? String(index + 1)
          : (keyForEnumOptionIndex(index) ?? String(index + 1)),
        label: enumLabels[index] ?? String(value),
        active: value === currentValue,
      })),
    );
    // In >10 mode the typed option number appears in the entry box on the nav
    // line (next to the prev/next hint), not among the option chips.
    if (useNumberEntry) {
      this.entry.render(entrySlot);
    }
  }

  protected onAnnotationChanged() {
    this.entry.reset();
  }

  protected navHintSuffix() {
    return this.useNumberEntry ? "  ·  Enter to set · Esc to exit" : "";
  }

  toJSON() {
    return annotateEnumPropertyToolJson(this.propertyIdentifier);
  }
}

// Bool property: an instant-action tool that toggles the value of the
// currently-selected annotation and immediately deactivates (unlike the enum
// and number tools, there is no data-entry mode).
class ToggleBoolPropertyTool extends LayerTool<AnnotationUserLayer> {
  constructor(
    public propertyIdentifier: string,
    layer: AnnotationUserLayer,
  ) {
    super(layer);
  }

  activate(activation: ToolActivation<this>) {
    updateSelectedAnnotationProperty(
      this.layer,
      this.propertyIdentifier,
      (currentValue) => (currentValue ? 0 : 1),
    );
    activation.cancel();
  }

  toJSON() {
    return toggleBoolPropertyToolJson(this.propertyIdentifier);
  }

  get description() {
    return `toggle ${this.propertyIdentifier}`;
  }
}

// Plain numeric property: type a number and press Enter to commit it, using the
// key-capture buffer (only number keys are routed; other keybinds stay live).
class NumberPropertyEntryTool extends AnnotationPropertyEntryTool {
  protected readonly modeVerb = "Set";

  private get isFloat(): boolean {
    return this.property?.type === "float32";
  }

  private get dataType(): DataType | undefined {
    const type = this.property?.type;
    return type === undefined ? undefined : propertyTypeDataType[type];
  }

  // The value range allowed by the data type. float32 uses its full magnitude
  // range rather than the normalized [0, 1] default.
  private range(): [number, number] {
    const { dataType } = this;
    if (dataType === undefined) return [-Infinity, Infinity];
    if (dataType === DataType.FLOAT32) {
      return [-3.40282347e38, 3.40282347e38];
    }
    return defaultDataTypeRange[dataType] as [number, number];
  }

  private readonly entry = new ToolValueEntryBuffer({
    isFloat: () => this.isFloat,
    allowNegative: () => this.range()[0] < 0,
    range: () => this.range(),
    currentValue: () => this.currentValue,
    commit: (value) =>
      updateSelectedAnnotationProperty(
        this.layer,
        this.propertyIdentifier,
        () => (this.isFloat ? value : Math.round(value)),
      ),
  });

  protected configureBindings(
    manager: ToolBindingManager,
    activation: ToolActivation<this>,
    requestRefresh: () => void,
  ) {
    manager.addGroup({
      id: "number",
      bindings: this.entry.bindings({
        requestRefresh,
        cancel: () => activation.cancel(),
        actionPrefix: "annotation-number",
      }),
    });
  }

  protected renderValue(valueContainer: HTMLElement, entrySlot: HTMLElement) {
    removeChildren(valueContainer);
    removeChildren(entrySlot);
    this.entry.render(entrySlot);
  }

  protected onAnnotationChanged() {
    this.entry.reset();
  }

  protected navHintSuffix() {
    return "  ·  Enter to set · Esc to exit";
  }

  toJSON() {
    return annotateNumberPropertyToolJson(this.propertyIdentifier);
  }
}

const Base = UserLayerWithAnnotationsMixin(UserLayer);

// A keybindable tool that selects the previous annotation in the iteration order.
class SelectPreviousAnnotationTool extends LayerTool<AnnotationUserLayer> {
  activate(activation: ToolActivation<this>) {
    this.layer.shiftSelectedIndexBy(-1);
    activation.cancel();
  }
  toJSON() {
    return SELECT_PREVIOUS_ANNOTATION_TOOL_ID;
  }
  get description() {
    return "select previous annotation";
  }
}

// A keybindable tool that selects the next annotation in the iteration order.
class SelectNextAnnotationTool extends LayerTool<AnnotationUserLayer> {
  activate(activation: ToolActivation<this>) {
    this.layer.shiftSelectedIndexBy(1);
    activation.cancel();
  }
  toJSON() {
    return SELECT_NEXT_ANNOTATION_TOOL_ID;
  }
  get description() {
    return "select next annotation";
  }
}

export class AnnotationUserLayer extends Base {
  localAnnotations: LocalAnnotationSource | undefined;
  codeVisible = new TrackableBoolean(true);
  hideInactiveShaderControls = new TrackableBoolean(false);
  // Accessed by the property keybind tools (e.g. EnumClassificationTool) to
  // observe schema changes while active.
  readonly localAnnotationProperties: WatchableValue<AnnotationPropertySpec[]> =
    new WatchableValue([]);
  private localAnnotationRelationships: string[];
  private localAnnotationsJson: any = undefined;
  // Keybindable property tools currently registered for this layer, keyed by
  // tool id. Covers both enum-value setters and bool toggles; the descriptor
  // records which kind each is so the tool can be reconstructed.
  private registeredPropertyTools = new Map<string, PropertyToolDescriptor>();
  private pointAnnotationsJson: any = undefined;
  static supportColorPickerInAnnotationTab = false;

  linkedSegmentationLayers = this.registerDisposer(
    new LinkedSegmentationLayers(
      this.manager.rootLayers,
      this.annotationStates,
      this.annotationDisplayState,
    ),
  );

  disposed() {
    const { localAnnotations } = this;
    if (localAnnotations !== undefined) {
      localAnnotations.dispose();
    }
    for (const toolId of this.registeredPropertyTools.keys()) {
      releasePropertyTool(toolId);
    }
    this.registeredPropertyTools.clear();
    super.disposed();
  }

  constructor(managedLayer: Borrowed<ManagedUserLayer>) {
    super(managedLayer);
    this.linkedSegmentationLayers.changed.add(
      this.specificationChanged.dispatch,
    );
    this.codeVisible.changed.add(this.specificationChanged.dispatch);
    this.hideInactiveShaderControls.changed.add(
      this.specificationChanged.dispatch,
    );
    this.annotationDisplayState.ignoreNullSegmentFilter.changed.add(
      this.specificationChanged.dispatch,
    );
    this.annotationCrossSectionRenderScaleTarget.changed.add(
      this.specificationChanged.dispatch,
    );
    this.annotationProjectionRenderScaleTarget.changed.add(
      this.specificationChanged.dispatch,
    );
    this.tabs.add("rendering", {
      label: "Rendering",
      order: -100,
      getter: () => new RenderingOptionsTab(this),
    });
    this.tabs.add("schema", {
      label: "Schema",
      order: 20,
      getter: () => new AnnotationSchemaTab(this),
    });
    this.tabs.default = "annotations";
    this.registerDisposer(
      this.localAnnotationProperties.changed.add(() => {
        this.syncPropertyTools();
      }),
    );
  }

  // Keep one keybindable data-entry tool registered per property (enum, bool,
  // or plain number) so each can be bound to a hotkey in the schema editor.
  // Registers newly-added properties and tears down removed ones, including any
  // live keybindings in this layer's tool binder.
  private syncPropertyTools(
    properties: readonly AnnotationPropertySpec[] = this
      .localAnnotationProperties.value,
  ) {
    const desired = new Map<string, PropertyToolDescriptor>();
    for (const property of properties) {
      const { identifier } = property;
      if (property.type === "bool") {
        desired.set(toggleBoolPropertyToolJson(identifier), {
          kind: "bool",
          propertyIdentifier: identifier,
        });
      } else if (isAnnotationNumericPropertySpec(property)) {
        if (property.enumValues !== undefined) {
          desired.set(annotateEnumPropertyToolJson(identifier), {
            kind: "enum",
            propertyIdentifier: identifier,
          });
        } else {
          desired.set(annotateNumberPropertyToolJson(identifier), {
            kind: "number",
            propertyIdentifier: identifier,
          });
        }
      }
    }
    // Tear down tools whose property no longer exists, including any live
    // keybinding for them.
    for (const toolId of this.registeredPropertyTools.keys()) {
      if (!desired.has(toolId)) {
        this.toolBinder.removeJsonString(JSON.stringify(toolId));
        releasePropertyTool(toolId);
        this.registeredPropertyTools.delete(toolId);
      }
    }
    // Register newly-added tools.
    for (const [toolId, descriptor] of desired) {
      if (!this.registeredPropertyTools.has(toolId)) {
        retainPropertyTool(toolId, descriptor);
        this.registeredPropertyTools.set(toolId, descriptor);
      }
    }
  }

  restoreState(specification: any) {
    const properties = verifyOptionalObjectProperty(
      specification,
      ANNOTATION_PROPERTIES_JSON_KEY,
      parseAnnotationPropertySpecs,
    );
    // Register property tools before `super.restoreState` restores the tool
    // binder, so that any saved property keybindings resolve to a registered
    // tool.
    this.syncPropertyTools(properties ?? []);
    super.restoreState(specification);
    this.linkedSegmentationLayers.restoreState(specification);
    this.codeVisible.restoreState(specification[CODE_VISIBLE_KEY]);
    this.hideInactiveShaderControls.restoreState(
      specification[HIDE_INACTIVE_SHADER_CONTROLS_JSON_KEY],
    );
    this.localAnnotationsJson = specification[ANNOTATIONS_JSON_KEY];
    this.localAnnotationProperties.value = properties ?? [];

    this.localAnnotationRelationships = verifyOptionalObjectProperty(
      specification,
      ANNOTATION_RELATIONSHIPS_JSON_KEY,
      verifyStringArray,
      ["segments"],
    );
    this.pointAnnotationsJson = specification[POINTS_JSON_KEY];
    this.annotationCrossSectionRenderScaleTarget.restoreState(
      specification[CROSS_SECTION_RENDER_SCALE_JSON_KEY],
    );
    this.annotationProjectionRenderScaleTarget.restoreState(
      specification[PROJECTION_RENDER_SCALE_JSON_KEY],
    );
    this.annotationDisplayState.ignoreNullSegmentFilter.restoreState(
      specification[IGNORE_NULL_SEGMENT_FILTER_JSON_KEY],
    );
    this.annotationDisplayState.shader.restoreState(
      specification[SHADER_JSON_KEY],
    );
    this.annotationDisplayState.shaderControls.restoreState(
      specification[SHADER_CONTROLS_JSON_KEY],
    );
  }

  getLegacyDataSourceSpecifications(
    sourceSpec: any,
    layerSpec: any,
    legacyTransform: CoordinateTransformSpecification | undefined,
    explicitSpecs: DataSourceSpecification[],
  ): DataSourceSpecification[] {
    if (Object.prototype.hasOwnProperty.call(layerSpec, "source")) {
      return super.getLegacyDataSourceSpecifications(
        sourceSpec,
        layerSpec,
        legacyTransform,
        explicitSpecs,
      );
    }
    const scales = verifyOptionalObjectProperty(
      layerSpec,
      "voxelSize",
      (voxelSizeObj) =>
        parseFixedLengthArray(
          new Float64Array(3),
          voxelSizeObj,
          (x) => verifyFinitePositiveFloat(x) / 1e9,
        ),
    );
    const units = ["m", "m", "m"];
    if (scales !== undefined) {
      const inputSpace = makeCoordinateSpace({
        rank: 3,
        units,
        scales,
        names: ["x", "y", "z"],
      });
      if (legacyTransform === undefined) {
        legacyTransform = {
          outputSpace: inputSpace,
          sourceRank: 3,
          transform: undefined,
          inputSpace,
        };
      } else {
        legacyTransform = {
          ...legacyTransform,
          inputSpace,
        };
      }
    }
    return [
      {
        url: localAnnotationsUrl,
        transform: legacyTransform,
        enableDefaultSubsources: true,
        subsources: new Map(),
      },
    ];
  }

  activateDataSubsources(subsources: Iterable<LoadedDataSubsource>) {
    let hasLocalAnnotations = false;
    let properties:
      | WatchableValue<readonly Readonly<AnnotationPropertySpec>[]>
      | undefined;
    for (const loadedSubsource of subsources) {
      const { subsourceEntry } = loadedSubsource;
      const { local } = subsourceEntry.subsource;
      const setProperties = (
        newProperties: WatchableValue<
          readonly Readonly<AnnotationPropertySpec>[]
        >,
      ) => {
        if (
          properties !== undefined &&
          stableStringify(newProperties.value) !==
            stableStringify(properties.value)
        ) {
          loadedSubsource.deactivate(
            "Annotation properties are not compatible",
          );
          return false;
        }
        properties = newProperties;
        return true;
      };
      if (local === LocalDataSource.annotations) {
        if (hasLocalAnnotations) {
          loadedSubsource.deactivate(
            "Only one local annotations source per layer is supported",
          );
          continue;
        }
        hasLocalAnnotations = true;
        if (!setProperties(this.localAnnotationProperties)) continue;
        loadedSubsource.activate((refCounted) => {
          const localAnnotations = (this.localAnnotations =
            new LocalAnnotationSource(
              loadedSubsource.loadedDataSource.transform,
              this.localAnnotationProperties,
              this.localAnnotationRelationships,
            ));
          try {
            localAnnotations.restoreState(this.localAnnotationsJson);
          } catch {
            // Ignore errors from malformed local annotations.
          }
          refCounted.registerDisposer(() => {
            localAnnotations.dispose();
            this.localAnnotations = undefined;
          });
          refCounted.registerDisposer(
            this.localAnnotations.changed.add(
              this.specificationChanged.dispatch,
            ),
          );
          try {
            addPointAnnotations(
              this.localAnnotations,
              this.pointAnnotationsJson,
            );
          } catch {
            // Ignore errors from malformed point annotations.
          }
          this.pointAnnotationsJson = undefined;
          this.localAnnotationsJson = undefined;
          const state = new AnnotationLayerState({
            localPosition: this.localPosition,
            transform: refCounted.registerDisposer(
              getWatchableRenderLayerTransform(
                this.manager.root.coordinateSpace,
                this.localPosition.coordinateSpace,
                loadedSubsource.loadedDataSource.transform,
                undefined,
              ),
            ),
            source: localAnnotations.addRef(),
            displayState: this.annotationDisplayState,
            dataSource: loadedSubsource.loadedDataSource.layerDataSource,
            subsourceIndex: loadedSubsource.subsourceIndex,
            subsourceId: subsourceEntry.id,
            role: RenderLayerRole.ANNOTATION,
          });
          this.addAnnotationLayerState(state, loadedSubsource);
        });
        continue;
      }
      const { annotation } = subsourceEntry.subsource;
      if (annotation !== undefined) {
        if (!setProperties(annotation.properties)) continue;
        loadedSubsource.activate(() => {
          const state = new AnnotationLayerState({
            localPosition: this.localPosition,
            transform: loadedSubsource.getRenderLayerTransform(),
            source: annotation,
            displayState: this.annotationDisplayState,
            dataSource: loadedSubsource.loadedDataSource.layerDataSource,
            subsourceIndex: loadedSubsource.subsourceIndex,
            subsourceId: subsourceEntry.id,
            role: RenderLayerRole.ANNOTATION,
          });
          this.addAnnotationLayerState(state, loadedSubsource);
        });
        continue;
      }
      loadedSubsource.deactivate("Not compatible with annotation layer");
    }
    const prevAnnotationProperties =
      this.annotationDisplayState.annotationProperties.value;
    if (
      properties !== undefined &&
      stableStringify(prevAnnotationProperties) !==
        stableStringify(properties.value)
    ) {
      this.registerDisposer(
        properties.changed.add(() => {
          this.annotationDisplayState.annotationProperties.value =
            properties !== undefined ? [...properties.value] : [];
        }),
      );
      this.annotationDisplayState.annotationProperties.value = [
        ...properties.value,
      ];
    }
  }

  initializeAnnotationLayerViewTab(tab: AnnotationLayerView) {
    const hasChunkedSource = tab.registerDisposer(
      makeCachedLazyDerivedWatchableValue(
        (states) =>
          states.some((x) => x.source instanceof MultiscaleAnnotationSource),
        this.annotationStates,
      ),
    );
    const renderScaleControls = tab.registerDisposer(
      new DependentViewWidget(
        hasChunkedSource,
        (hasChunkedSource, parent, refCounted) => {
          if (!hasChunkedSource) return;
          {
            const renderScaleWidget = refCounted.registerDisposer(
              new RenderScaleWidget(
                this.annotationCrossSectionRenderScaleHistogram,
                this.annotationCrossSectionRenderScaleTarget,
              ),
            );
            renderScaleWidget.label.textContent = "Spacing (cross section)";
            parent.appendChild(renderScaleWidget.element);
          }
          {
            const renderScaleWidget = refCounted.registerDisposer(
              new RenderScaleWidget(
                this.annotationProjectionRenderScaleHistogram,
                this.annotationProjectionRenderScaleTarget,
              ),
            );
            renderScaleWidget.label.textContent = "Spacing (projection)";
            parent.appendChild(renderScaleWidget.element);
          }
        },
      ),
    );
    tab.element.insertBefore(
      renderScaleControls.element,
      tab.element.firstChild,
    );
    {
      const checkbox = tab.registerDisposer(
        new TrackableBooleanCheckbox(
          this.annotationDisplayState.ignoreNullSegmentFilter,
        ),
      );
      const label = document.createElement("label");
      label.appendChild(
        document.createTextNode("Ignore null related segment filter"),
      );
      label.title =
        "Display all annotations if filtering by related segments is enabled but no segments are selected";
      label.appendChild(checkbox.element);
      tab.element.appendChild(label);
    }
    tab.element.appendChild(
      tab.registerDisposer(
        new LinkedSegmentationLayersWidget(this.linkedSegmentationLayers),
      ).element,
    );
  }

  toJSON() {
    const x = super.toJSON();
    x[CROSS_SECTION_RENDER_SCALE_JSON_KEY] =
      this.annotationCrossSectionRenderScaleTarget.toJSON();
    x[CODE_VISIBLE_KEY] = this.codeVisible.toJSON();
    x[HIDE_INACTIVE_SHADER_CONTROLS_JSON_KEY] =
      this.hideInactiveShaderControls.toJSON();
    x[PROJECTION_RENDER_SCALE_JSON_KEY] =
      this.annotationProjectionRenderScaleTarget.toJSON();
    if (this.localAnnotations !== undefined) {
      x[ANNOTATIONS_JSON_KEY] = this.localAnnotations.toJSON();
    } else if (this.localAnnotationsJson !== undefined) {
      x[ANNOTATIONS_JSON_KEY] = this.localAnnotationsJson;
    }
    x[ANNOTATION_PROPERTIES_JSON_KEY] = annotationPropertySpecsToJson(
      this.localAnnotationProperties.value,
    );
    const { localAnnotationRelationships } = this;
    x[ANNOTATION_RELATIONSHIPS_JSON_KEY] =
      localAnnotationRelationships.length === 1 &&
      localAnnotationRelationships[0] === "segments"
        ? undefined
        : localAnnotationRelationships;
    x[IGNORE_NULL_SEGMENT_FILTER_JSON_KEY] =
      this.annotationDisplayState.ignoreNullSegmentFilter.toJSON();
    x[SHADER_JSON_KEY] = this.annotationDisplayState.shader.toJSON();
    x[SHADER_CONTROLS_JSON_KEY] =
      this.annotationDisplayState.shaderControls.toJSON();
    Object.assign(x, this.linkedSegmentationLayers.toJSON());
    return x;
  }

  observeLayerColor(callback: () => void) {
    const disposer = super.observeLayerColor(callback);
    const subDisposer = observeWatchable(
      callback,
      this.annotationDisplayState.color,
    );
    const shaderDisposer = observeWatchable(
      callback,
      this.annotationDisplayState.shader,
    );
    return () => {
      disposer();
      subDisposer();
      shaderDisposer();
    };
  }

  get automaticLayerBarColors() {
    const shaderHasDefaultColor =
      this.annotationDisplayState.shader.value.includes("defaultColor");
    if (shaderHasDefaultColor && this.annotationDisplayState.color.value) {
      const [r, g, b] = this.annotationDisplayState.color.value;
      return [`rgb(${r * 255}, ${g * 255}, ${b * 255})`];
    }

    return undefined;
  }

  static type = "annotation";
  static typeAbbreviation = "ann";
  static supportsLayerBarColorSyncOption = true;
}

function makeShaderCodeWidget(layer: AnnotationUserLayer) {
  return new ShaderCodeWidget({
    shaderError: layer.annotationDisplayState.shaderError,
    fragmentMain: layer.annotationDisplayState.shader,
    shaderControlState: layer.annotationDisplayState.shaderControls,
  });
}

class ShaderCodeOverlay extends Overlay {
  codeWidget: ShaderCodeWidget;
  constructor(public layer: AnnotationUserLayer) {
    super();
    this.codeWidget = this.registerDisposer(makeShaderCodeWidget(this.layer));
    this.content.appendChild(this.codeWidget.element);
    this.codeWidget.textEditor.refresh();
  }
}

class RenderingOptionsTab extends Tab {
  codeWidget: ShaderCodeWidget;
  constructor(public layer: AnnotationUserLayer) {
    super();
    const { element } = this;
    this.codeWidget = this.registerDisposer(makeShaderCodeWidget(this.layer));
    element.classList.add("neuroglancer-annotation-rendering-tab");
    const shaderProperties = this.registerDisposer(
      new DependentViewWidget(
        layer.annotationDisplayState.annotationProperties,
        (properties, parent) => {
          if (properties === undefined || properties.length === 0) return;
          buildShaderPropertyList(properties, parent);
        },
      ),
    ).element;

    layer.registerDisposer(
      new ElementVisibilityFromTrackableBoolean(
        layer.codeVisible,
        shaderProperties,
      ),
    );

    element.appendChild(shaderProperties);
    element.appendChild(
      makeShaderCodeWidgetTopRow(
        this.layer,
        this.codeWidget,
        ShaderCodeOverlay,
        {
          title: "Documentation on image layer rendering",
          href: "https://github.com/google/neuroglancer/blob/master/src/annotation/rendering.md",
        },
        "neuroglancer-annotation-dropdown-shader-top-row",
      ),
    );

    element.appendChild(this.codeWidget.element);
    element.appendChild(
      this.registerDisposer(
        new ShaderControls(
          layer.annotationDisplayState.shaderControls,
          this.layer.manager.root.display,
          this.layer,
          {
            visibility: this.visibility,
            hideInactiveShaderControls: layer.hideInactiveShaderControls,
          },
        ),
      ).element,
    );

    element.appendChild(
      addLayerControlToOptionsTab(
        this,
        layer,
        this.visibility,
        LAYER_CONTROLS[ANNOTATION_COLOR_JSON_KEY],
      ),
    );
  }
}

const LAYER_CONTROLS: Record<
  string,
  LayerControlDefinition<AnnotationUserLayer>
> = {
  [ANNOTATION_COLOR_JSON_KEY]: {
    label: "Annotation color",
    title:
      "Annotation shader default color (enabled with 'defaultColor()' in the shader code)",
    toolJson: ANNOTATION_COLOR_JSON_KEY,
    ...colorLayerControl(
      (layer: AnnotationUserLayer) => layer.annotationDisplayState.color,
    ),
    isValid: (layer) =>
      makeCachedLazyDerivedWatchableValue(
        (shader) => shader.match(/\bdefaultColor\b/) !== null,
        layer.annotationDisplayState.shaderControls.processedFragmentMain,
      ),
  },
};

for (const control of Object.values(LAYER_CONTROLS)) {
  registerLayerControl(AnnotationUserLayer, control);
}

registerLayerType(AnnotationUserLayer);
registerLayerType(AnnotationUserLayer, "pointAnnotation");

registerTool(
  AnnotationUserLayer,
  SELECT_PREVIOUS_ANNOTATION_TOOL_ID,
  (layer) => new SelectPreviousAnnotationTool(layer),
);
registerTool(
  AnnotationUserLayer,
  SELECT_NEXT_ANNOTATION_TOOL_ID,
  (layer) => new SelectNextAnnotationTool(layer),
);

registerLayerTypeDetector((subsource) => {
  if (subsource.local === LocalDataSource.annotations) {
    return { layerConstructor: AnnotationUserLayer, priority: 100 };
  }
  if (subsource.annotation !== undefined) {
    return { layerConstructor: AnnotationUserLayer, priority: 1 };
  }
  return undefined;
});

registerLayerShaderControlsTool(AnnotationUserLayer, (layer) => ({
  shaderControlState: layer.annotationDisplayState.shaderControls,
}));
