# @license
# Copyright 2020 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pytest
from neuroglancer import viewer_state


def test_coordinate_space_from_json():
    x = viewer_state.CoordinateSpace(
        {
            "x": [4e-9, "m"],
            "y": [5e-9, "m"],
            "z": [6e-9, "m"],
            "t": [2, "s"],
        }
    )
    assert x.names == ("x", "y", "z", "t")
    np.testing.assert_array_equal(x.scales, [4e-9, 5e-9, 6e-9, 2])
    assert x.units == ("m", "m", "m", "s")
    assert x.rank == 4
    assert x[0] == viewer_state.DimensionScale(4e-9, "m")
    assert x[0:2] == [
        viewer_state.DimensionScale(4e-9, "m"),
        viewer_state.DimensionScale(5e-9, "m"),
    ]
    assert x["x"] == viewer_state.DimensionScale(4e-9, "m")
    assert x[1] == viewer_state.DimensionScale(5e-9, "m")
    assert x["y"] == viewer_state.DimensionScale(5e-9, "m")
    assert x[2] == viewer_state.DimensionScale(6e-9, "m")
    assert x["z"] == viewer_state.DimensionScale(6e-9, "m")
    assert x[3] == viewer_state.DimensionScale(2, "s")
    assert x["t"] == viewer_state.DimensionScale(2, "s")
    assert x.to_json() == {
        "x": [4e-9, "m"],
        "y": [5e-9, "m"],
        "z": [6e-9, "m"],
        "t": [2, "s"],
    }


def test_coordinate_space_from_split():
    x = viewer_state.CoordinateSpace(
        names=["x", "y", "z", "t"], scales=[4, 5, 6, 2], units=["nm", "nm", "nm", "s"]
    )
    assert x.to_json() == {
        "x": [4e-9, "m"],
        "y": [5e-9, "m"],
        "z": [6e-9, "m"],
        "t": [2, "s"],
    }


def test_layers():
    layer_json = [
        {"name": "a", "type": "segmentation", "visible": False},
        {"name": "b", "type": "image"},
    ]
    layers_ro = viewer_state.Layers(layer_json, _readonly=True)
    assert layers_ro[0].name == "a"
    assert isinstance(layers_ro[0].layer, viewer_state.SegmentationLayer)
    assert layers_ro[0].visible is False
    assert isinstance(layers_ro["a"].layer, viewer_state.SegmentationLayer)
    assert layers_ro[1].name == "b"
    assert isinstance(layers_ro[1].layer, viewer_state.ImageLayer)
    assert layers_ro[1].visible is True
    assert isinstance(layers_ro["b"].layer, viewer_state.ImageLayer)

    with pytest.raises(AttributeError):
        layers_ro["c"] = viewer_state.ImageLayer()

    with pytest.raises(AttributeError):
        del layers_ro[0]
    with pytest.raises(AttributeError):
        del layers_ro["a"]
    with pytest.raises(AttributeError):
        del layers_ro[:]

    layers_rw = viewer_state.Layers(layer_json)
    del layers_rw[0]
    assert layers_rw.to_json() == [
        {"name": "b", "type": "image"},
    ]

    layers_rw = viewer_state.Layers(layer_json)
    del layers_rw["a"]
    assert layers_rw.to_json() == [
        {"name": "b", "type": "image"},
    ]

    layers_rw = viewer_state.Layers(layer_json)
    del layers_rw[:]
    assert layers_rw.to_json() == []


def test_tool():
    p = viewer_state.Tool("shaderControl", control="abc")
    assert isinstance(p, viewer_state.ShaderControlTool)
    assert p.control == "abc"

    p2 = viewer_state.ShaderControlTool(control="abc")
    assert p2.control == "abc"


ANNOTATION_PROPERTY_TOOLS = [
    ("toggleBoolProperty", "reviewed", viewer_state.ToggleBoolPropertyTool),
    ("annotateEnumProperty", "status", viewer_state.AnnotateEnumPropertyTool),
    ("annotateNumberProperty", "score", viewer_state.AnnotateNumberPropertyTool),
]

ANNOTATION_NAV_TOOLS = [
    ("selectPreviousAnnotation", viewer_state.SelectPreviousAnnotationTool),
    ("selectNextAnnotation", viewer_state.SelectNextAnnotationTool),
]


@pytest.mark.parametrize("tool_type,prop,cls", ANNOTATION_PROPERTY_TOOLS)
def test_annotation_property_tool(tool_type, prop, cls):
    tool = cls(property=prop)
    assert tool.to_json() == {"type": tool_type, "property": prop}
    assert tool.property == prop

    from_json = viewer_state.Tool({"type": tool_type, "property": prop})
    assert isinstance(from_json, cls)
    assert from_json.property == prop


@pytest.mark.parametrize("tool_type,cls", ANNOTATION_NAV_TOOLS)
def test_annotation_navigation_tool(tool_type, cls):
    assert cls().to_json() == {"type": tool_type}
    assert isinstance(viewer_state.Tool(tool_type), cls)
    # Neuroglancer serializes these as a bare string and normalizes a bare
    # string back to object form on restore, so emitting the object form (as
    # every other Python tool does) round-trips correctly.
    assert viewer_state.Tool(tool_type).to_json() == {"type": tool_type}


def test_annotation_property_tool_bindings_from_state():
    """Regression test: this raised ``KeyError`` before the tools were registered."""
    state = viewer_state.ViewerState(
        {
            "layers": [
                {
                    "name": "a",
                    "type": "annotation",
                    "source": "local://annotations",
                    "annotationProperties": [
                        {"id": "reviewed", "type": "bool"},
                        {
                            "id": "status",
                            "type": "uint8",
                            "enum_values": [0, 1, 2],
                            "enum_labels": ["unknown", "good", "bad"],
                        },
                        {"id": "score", "type": "float32"},
                    ],
                    "toolBindings": {
                        "R": {"type": "toggleBoolProperty", "property": "reviewed"},
                        "S": {"type": "annotateEnumProperty", "property": "status"},
                        "E": {"type": "annotateNumberProperty", "property": "score"},
                        "P": "selectPreviousAnnotation",
                        "N": "selectNextAnnotation",
                    },
                }
            ]
        }
    )
    bindings = state.layers["a"].tool_bindings
    assert isinstance(bindings["R"], viewer_state.ToggleBoolPropertyTool)
    assert bindings["R"].property == "reviewed"
    assert isinstance(bindings["S"], viewer_state.AnnotateEnumPropertyTool)
    assert bindings["S"].property == "status"
    assert isinstance(bindings["E"], viewer_state.AnnotateNumberPropertyTool)
    assert bindings["E"].property == "score"
    assert isinstance(bindings["P"], viewer_state.SelectPreviousAnnotationTool)
    assert isinstance(bindings["N"], viewer_state.SelectNextAnnotationTool)


def test_unknown_tool_type_is_opaque():
    state = viewer_state.ViewerState(
        {
            "toolBindings": {
                "A": {"type": "futureTool", "extra": 1},
                "B": "selectNextAnnotation",
            }
        }
    )
    with pytest.warns(UserWarning, match="Unknown tool type"):
        bindings = state.tool_bindings
    # Unrecognized, but round-trips exactly, including unknown fields.
    assert type(bindings["A"]) is viewer_state.Tool
    assert bindings["A"].type == "futureTool"
    assert bindings["A"].to_json() == {"type": "futureTool", "extra": 1}
    # A sibling binding in the same map still resolves.
    assert isinstance(bindings["B"], viewer_state.SelectNextAnnotationTool)


def test_invalid_tool_type():
    with pytest.raises(ValueError, match="Unknown tool type"):
        viewer_state.Tool({})
    with pytest.raises(ValueError, match="Unknown tool type"):
        viewer_state.Tool({"type": 5})


@pytest.mark.parametrize("key", ["a", "AB", "1", "", "keyq"])
def test_invalid_tool_binding_key(key):
    bindings = viewer_state.ToolBindings()
    with pytest.raises(ValueError, match="Invalid tool binding key"):
        bindings[key] = viewer_state.SelectNextAnnotationTool()


def test_tool_binding_key_validation_is_write_only():
    bindings = viewer_state.ToolBindings()
    bindings["R"] = viewer_state.SelectNextAnnotationTool()
    assert bindings.to_json() == {"R": {"type": "selectNextAnnotation"}}

    # Reading back a state written by another client stays permissive.
    state = viewer_state.ViewerState({"toolBindings": {"a": "selectNextAnnotation"}})
    assert list(state.tool_bindings.keys()) == ["a"]


def test_layer_tool_bindings_alias():
    layer = viewer_state.Layer(
        toolBindings={"R": viewer_state.SelectNextAnnotationTool()}
    )
    assert layer.tool_bindings.to_json() == {"R": {"type": "selectNextAnnotation"}}


@pytest.mark.parametrize("property_type", viewer_state.ANNOTATION_PROPERTY_TYPES)
def test_annotation_property_type_valid(property_type):
    spec = viewer_state.AnnotationPropertySpec(id="a", type=property_type)
    assert spec.type == property_type


@pytest.mark.parametrize("property_type", ["float64", "string", "uint64"])
def test_annotation_property_type_validation_is_write_only(property_type):
    with pytest.raises(ValueError, match="Invalid annotation property type"):
        viewer_state.AnnotationPropertySpec(id="a", type=property_type)

    spec = viewer_state.AnnotationPropertySpec(id="a", type="uint8")
    with pytest.raises(ValueError, match="Invalid annotation property type"):
        spec.type = property_type

    # Parsing an existing spec stays permissive, so odd on-disk precomputed
    # `info` files remain readable.
    parsed = viewer_state.AnnotationPropertySpec({"id": "a", "type": property_type})
    assert parsed.type == property_type


def test_annotation_property_spec_tool():
    bool_spec = viewer_state.AnnotationPropertySpec(id="reviewed", type="bool")
    assert isinstance(bool_spec.tool(), viewer_state.ToggleBoolPropertyTool)
    assert bool_spec.tool().property == "reviewed"

    enum_spec = viewer_state.AnnotationPropertySpec(
        id="status", type="uint8", enum_values=[0, 1], enum_labels=["a", "b"]
    )
    assert isinstance(enum_spec.tool(), viewer_state.AnnotateEnumPropertyTool)

    number_spec = viewer_state.AnnotationPropertySpec(id="score", type="float32")
    assert isinstance(number_spec.tool(), viewer_state.AnnotateNumberPropertyTool)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"id": "tint", "type": "rgb"}, "No annotation property tool"),
        ({"id": "tint", "type": "rgba"}, "No annotation property tool"),
        (
            {
                "id": "status",
                "type": "uint8",
                "enum_values": [0, 1],
                "enum_labels": ["a"],
            },
            "same length as enum_values",
        ),
        (
            {"id": "status", "type": "uint8", "enum_labels": ["a"]},
            "enum_labels without enum_values",
        ),
    ],
)
def test_annotation_property_spec_tool_invalid(kwargs, match):
    spec = viewer_state.AnnotationPropertySpec(**kwargs)
    with pytest.raises(ValueError, match=match):
        spec.tool()


def test_annotation():
    viewer_state.PointAnnotation(point=[1])
