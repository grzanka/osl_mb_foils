# MBO Workflow History

## 2026-03-26: Post-alignment background subtraction exploration

### User prompt

Add another step after align_ccb for 03-17 and 03-19 foils. In this step I would like to explore the background subtraction method which may or may not combine these two datasets.

Take as the input the data produced by align_ccb step.

First approach, call it scenario_A you should: subtract raw signal "foil1 from 03-19" - "foil1 from 03-17", present such raw signal with configurable isolines. When subtracting ensure that the circles for both foils are centered around the same place on both images.
Then do the same for foil2.

Do not subtract or present any LV images.

In scenario B apply take an area from center to 80% of radius for foils 3 and 4 from 03-17 dataset. Calculate mean value from this area from both foils, then take average from these two values. This should be a fixed number to be used as a background level.

Then subtract this number from all raw values for foil1 and foil2 in 03-17 and all foils from 03-19 dataset. Present the results.

Then take the values from foil1 and foil2 (after background subtraction). Normalize them by dividing of mean value taken from area up to 90% of radius of their corresponding circles. After such normalization the values witin circles should be close to zero. Apply some aggresive smoothing, possibly gaussian 2D kernel to reduce noise. Present the results with proper isolines from 0.9 to 1.1.

In final stage divide foil1 from 03-19 (after background subtraction) by the normalized foil (from 03-17). Do the same for foil2 from 03-19.

Present all immediate steps in PDF, using group of respective foils per row. Include good descriptions and pairs of vertical+horizontal profiles.

Run the code with poetry to check if everything works properly.

Note down all your actions and my prompt in some markdown file in the most appropriate place to keep a written history of my prompting and your actions.

### Actions taken

- Added a new `mbo_background_subtraction` config type and CLI dispatch under the pub2026 MBO pipeline.
- Implemented `src/pub2026/mbo/background_subtraction.py` to consume aligned NPZ outputs and generate both requested scenarios.
- Added scenario A centered subtraction for foils 1 and 2, with configurable contour levels and horizontal/vertical profiles.
- Added scenario B fixed-background estimation from 03-17 foils 3 and 4, background subtraction pages, normalized 03-17 foil smoothing, and final ratio pages for foils 1 and 2.
- Added a dedicated YAML config for the 03-17 / 03-19 CCB pair.
- Added config and CLI coverage plus a synthetic end-to-end test for the new stage.
- Updated the pub2026 README so the new step is documented.
- Planned validation with `poetry run python -m src.pub2026.cli run ...` after code changes.

---

## 2026-03-26: PDF layout redesign — colorscale + page structure

### User prompt

In the generated PDF start with presenting PDF with scenario A output.
I would like to see on one page original foil data and the result of subtraction. When presenting subtraction result, the colorscale should start from zero. I do not want to see any negative values. Zero should be represented as white, maximum value as red and half of max value as green.

The horizontal/vertical profiles should be presented on same plot as two dataseries.

On first page in PDF I would like to see foils to be subtracted and result of subtraction. On second page: the result of subtraction and profiles.

Later I would like to get results for scenario B, clearly distinguished from scenario A.
Here, after background subtraction (which is a constant value) use same colorscale rule: I do not want to see any negative values. Zero should be represented as white, maximum value as red and half of max value as green.

Document my prompt and your actions.

### Actions taken

- Added `_WGR_CMAP` (white → green → red) custom `LinearSegmentedColormap` to `background_subtraction.py`.
- Added `_clamp_nonneg()` helper: clamps negative pixel values to zero before rendering.
- Added `_plot_profiles_combined()` helper: plots H and V profiles as two labelled series on one axes.
- Replaced `_plot_scenario_a_rows()` with two new page functions:
  - `_plot_scenario_a_page1()` — 3-column layout (reference | target | clamped diff with WGR).
  - `_plot_scenario_a_page2()` — 2-column layout (clamped diff with WGR | combined H+V profiles).
- Updated `_plot_bg_subtracted_rows()` to use WGR + `vmin=0`, combined H+V profiles (2-column).
- Updated `_plot_scenario_b_rows()` to use WGR for bg-subtracted images; combined profiles column (5-column layout).
- Restructured `explore_background_subtraction()` PDF page order: scenario A section first (header + page 1 + page 2), then scenario B section (header with background level info + background-source foils + bg-subtracted images + ratio figure + normalization details).
- Updated `scenario_a_contour_levels` to positive values only `[25, 50, 100, 200, 300]`.
- Verified: `poetry run pytest` — 34 passed; pipeline re-run produced updated PDF and NPZ.

---

## 2026-03-26: Scenario B division and colorscale alignment

### User prompt

In scenario B you should take foil 1 and 2 "bg -sub" from 03-13 and divide it by 03-17 foil 1 and 2 (respectively) "normalized x smoothed".

I see that the normalized foil have values somehow close to 0.9...1.1 range, therefore I wouln not expect that the result should be so different from what was before division.

If division by zero happens, put there as a result zero, instead of not-a-number.

Also - the result after division should be plotted exactly with same colorscale as before, keeping even sam max color value for the same foil number.

Document my prompt and your actions.

### Actions taken

- Kept scenario B as background-subtracted target foil divided by normalized×smoothed reference foil, matching the configured 03-17 and 03-19 datasets already used by this pipeline.
- Added `_safe_divide_with_zero()` so invalid divisions now default to `0.0` instead of `NaN`.
- Added `_positive_vmax()` so each scenario B ratio plot reuses the same per-foil `vmax` as the corresponding background-subtracted images.
- Updated `_plot_scenario_b_rows()` so the division result is rendered with the same white→green→red colorscale and `vmin=0` / shared `vmax`, instead of the previous ratio-centered `coolwarm` scale.
- Updated scenario B profile extraction to use the plotted division result, so the profile view now matches the image view.
- Added a regression test that forces zero division and verifies the stored ratio values are `0.0` wherever the division is invalid.
- Re-ran tests: `35 passed`.
- Re-ran the pipeline with Poetry and regenerated the scenario B PDF and NPZ outputs successfully.

---

## 2026-03-26: Additional Scenario B pages and Scenario A/B comparison

### User prompt

For scenario B:
Present on a separate page the "normalizedxsmoothed" with horintoal and vertical profiles.

Then on a next page present "bg-sub" 2D data, then "03/19 / norx smoothed" 2D data, then comparizon of vertical profiles from both 2D images, then comparison of vertical profiles from both images.

Finally add a page where scenario A is compared with scenario B, in terms of 2D raw signal maps and vertical and horizontal profiles.

Document my and your stuff as before.

### Actions taken

- Added `_plot_profile_comparison()` helper to overlay two profiles on one axes with consistent labeling.
- Added `_plot_scenario_b_normalized_rows()` to create a dedicated page for the normalized×smoothed images with combined horizontal and vertical profiles.
- Added `_plot_scenario_b_bgsub_vs_ratio_rows()` to create the next page with four columns per foil: bg-sub 2D image, divided-result 2D image, vertical-profile comparison, and horizontal-profile comparison.
- Interpreted the duplicated request for two vertical-profile comparisons as vertical plus horizontal profile comparisons, since that matches the surrounding requirement and produces a complete page.
- Added `_plot_scenario_ab_comparison_rows()` to compare Scenario A and Scenario B directly for each foil using 2D signal maps plus vertical and horizontal profile overlays.
- Replaced the previous single combined Scenario B page in the report with the new dedicated normalized page, the dedicated bg-sub vs divided-result page, and the final Scenario A vs Scenario B comparison page.
- Re-ran tests: `35 passed`.
- Re-ran the pipeline with Poetry and regenerated the PDF and NPZ outputs successfully.

---

## 2026-03-26: Workflow diagrams in PDF + per-foil target background subtraction

### User prompt

Document well steps for scenario A and B in generated PDF, you can use a diagram for that as well.

In scenario B add additional step, namely for foils from 19-03, instead of subtracting fixed number being a background signal, subtract mean value of signal from foils 03-17. For foil 1 from 03-19, instead of subtracting background level, subtract mean value from foil 1 from 03-17, from the area up to 90 percent of circle radius, but only from pixels which Y coordinate is higher than 7mm. The same should be done for foil 2.

Background should still be subtracted for foils 1 and from 03-17, to obtain normalized stuff. Rest of the steps should stay the same.

Document my steps as well.

### Actions taken

- Added `_draw_flowchart_scenario_a()` — creates a matplotlib-based flowchart diagram documenting the Scenario A workflow (load → center → subtract → clamp → output).
- Added `_draw_flowchart_scenario_b()` — creates a matplotlib-based flowchart diagram documenting the Scenario B workflow, with two branches showing:
  - Left branch: reference foils get a fixed background level (from foils 3, 4), then normalization and smoothing.
  - Right branch: target foils get a per-foil background level (mean of corresponding reference foil, filtered by radius fraction and Y threshold).
  - Both branches merge at the division step.
- Both diagrams are inserted as early pages in the PDF, right after the parameters page.
- Added `target_bg_y_threshold_mm` config parameter (default `7.0`) to `MBOBackgroundSubtractionConfig`.
- Added `_circle_mean_y_filtered()` helper: computes the mean value inside a circle radius fraction, restricted to pixels where the Y coordinate (relative to circle center) exceeds a configurable threshold in mm. Falls back to the regular `_circle_mean()` if no pixels pass the Y filter.
- Modified Scenario B in `explore_background_subtraction()`:
  - **Reference foils (03-17)**: continue to subtract the fixed background level (average of foils 3, 4 inside 80% radius). This is unchanged.
  - **Target foils (03-19)**: instead of subtracting the same fixed background level, each target foil now subtracts the mean of its corresponding reference foil, computed inside 90% radius and restricted to Y > 7 mm.
- Updated the Scenario B text page in the PDF to document both background subtraction approaches (fixed for reference, per-foil for target) with all numerical values.
- Updated PDF captions and descriptions to reflect the new per-foil background levels.
- Per-foil background levels are saved in the output NPZ as `target_foil_{id}_bg_level`.
- Existing tests pass unchanged (79 passed) because the synthetic test images are small enough that the Y threshold filter falls back gracefully to the unfiltered mean.
