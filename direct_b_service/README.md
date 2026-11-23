# Direct target service at the original registry size

This independent reference implementation runs the historical ridge readout on the target labels, with raw-threshold and group-held-out calibrated-threshold decision rules. The CSVs contain request-level mask predictions and aggregate counts. From the extracted archive run `python verify_budget.py` to recreate the inputs and reference outputs and replay the full budget curve.
