import argparse
import webbrowser

import neuroglancer
import neuroglancer.cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers["image"] = neuroglancer.ImageLayer(
            source="precomputed://gs://neuroglancer-public-data/flyem_fib-25/image",
            tool_bindings={
                "A": neuroglancer.ShaderControlTool(control="normalized"),
                "B": neuroglancer.OpacityTool(),
            },
        )

        # Annotation property tools: define a schema on a local annotation
        # layer, then bind a hotkey per property.  `AnnotationPropertySpec.tool`
        # picks the right tool for each property type.
        reviewed = neuroglancer.AnnotationPropertySpec(id="reviewed", type="bool")
        status = neuroglancer.AnnotationPropertySpec(
            id="status",
            type="uint8",
            enum_values=[0, 1, 2],
            enum_labels=["unknown", "good", "bad"],
        )
        score = neuroglancer.AnnotationPropertySpec(id="score", type="float32")
        s.layers["annotations"] = neuroglancer.LocalAnnotationLayer(
            dimensions=s.dimensions,
            annotation_properties=[reviewed, status, score],
            tool_bindings={
                "R": reviewed.tool(),
                "S": status.tool(),
                "E": score.tool(),
                "P": neuroglancer.SelectPreviousAnnotationTool(),
                "N": neuroglancer.SelectNextAnnotationTool(),
            },
        )

    print(viewer)
    webbrowser.open_new(viewer.get_viewer_url())
