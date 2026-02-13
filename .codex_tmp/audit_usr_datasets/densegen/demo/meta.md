name: densegen/demo
created_at: 2026-02-12T23:35:15.694011Z
source: audit
notes: post-registry init
schema: USR v1

### Updates (2026-02-12)
- 2026-02-12T23:35:15.694011Z: initialized dataset.
- 2026-02-12T23:35:16.553732Z: Imported 24 records from src/dnadesign/usr/demo_material/demo_sequences.csv
```bash
usr import densegen/demo --from csv --path src/dnadesign/usr/demo_material/demo_sequences.csv --bio-type dna --alphabet dna_4
```
- 2026-02-12T23:35:17.267724Z: Attached columns under 'quickstart' (24 row match)
```bash
usr attach densegen/demo --path src/dnadesign/usr/demo_material/demo_attachment_one.csv --namespace quickstart --key sequence --key-col sequence --columns "X_value"
```
- 2026-02-12T23:35:17.982745Z: Attached columns under 'quickstart' (24 row match)
```bash
usr attach densegen/demo --path src/dnadesign/usr/demo_material/demo_y_sfxi.csv --namespace quickstart --key sequence --key-col sequence --columns "intensity_log2_offset_delta" --allow-missing
```
- 2026-02-12T23:35:18.805978Z: Materialized overlays
```bash
usr materialize densegen/demo
```
