# ICBINB-BIO call for papers (fetched 2026-11-03)

The NeurIPS 2026 workshop in this series is **ICBINB-BIO**, “I Can't Believe It's Not Better (ICBINB): Failure Modes of AI in Biology,” not a page on icbinb.cc. The initiative site https://icbinb.cc/ still points at earlier workshops (including ICLR) and does not host this call. Facts below are copied from the pages named. Nothing that was not on those pages is filled in.

## Sources

- Call for papers: https://icbinb-bio.github.io/submit/
- Workshop home: https://icbinb-bio.github.io/
- Official style-file link on the CFP (“download style files”): https://drive.google.com/file/d/1YM_VG1bk6mokRHFwC69Z_nJnAJFvbHu0/view?usp=drive_link
- The zip contains only `neurips_2026.sty` and `neurips_2026.tex` (both dated 2026-07-22 in the archive listing). Those two files are the template used for `main.tex`. The style comment in `neurips_2026.tex` says full papers may be up to eight pages and tiny papers up to four, including figures.
- OpenReview portal linked from the CFP: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/ICBINB-BIO
- NeurIPS ethics guidelines linked from the CFP: https://neurips.cc/public/EthicsGuidelines
- NeurIPS 2026 workshop list naming this event: https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/
- A third-party EPFL notice (https://memento.epfl.ch/event/icbinb-bio-workshop-at-neurips-2026-call-for-paper/) still shows the pre-extension deadline and “December 11 or 12.” Where it disagrees with icbinb-bio.github.io, the workshop site is the source used here.

## Page limit and format

- **Full papers: up to eight pages, excluding references and appendices.** The CFP says this explicitly. The style file adds that pages containing acknowledgments, references, a checklist, and optional technical appendices do not count, and that papers exceeding the limit will not be reviewed. Figures count toward the eight pages.
- **Tiny papers: up to four pages of main text,** same formatting, and they need not contain all four required elements. This draft is a full paper, not a tiny paper.
- Appendices are unlimited. Reviewers are not required to read them.
- US Letter. The style file sets the text block. Do not edit `neurips_2026.sty`.
- Template option used for submission: `\usepackage[dblblindworkshop]{neurips_2026}` together with `\workshoptitle{...}`. The shell text says to omit `final` and `preprint` at submission time. `dblblindworkshop` leaves the anonymous flag on, prints “Anonymous Author(s),” and adds line numbers.

## Anonymity

- The CFP: “Submissions are double-blind; linked material must also preserve anonymity.”
- The style shell: cite one’s own published work in the third person. Do not put acknowledgments in the anonymized submission. The `ack` environment is hidden when the submission options are used.
- This draft therefore has no author names, no repository URL, and no acknowledgment.

## Deadline and other dates

All times below are as printed on the workshop site: **11:59 p.m. Anywhere on Earth.**

| Item | Date on https://icbinb-bio.github.io/submit/ |
| --- | --- |
| Submission | September 2, 2026 (August 29, 2026 is struck through) |
| Review window | September 2, 2026 – September 21, 2026 (the earlier window is struck through) |
| Notification | September 29, 2026 |
| Camera-ready and poster | November 11, 2026 |
| In-person workshop | December 11, 2026 |

The home page repeats the September 2, 2026 extension and shows camera-ready as November 11, 2026, with October 20, 2026 struck through. It also marks the camera-ready line and the statement “remain subject to final confirmation.” The home page adds the workshop hours **8:50 a.m.–5:00 p.m. Australia time** in Sydney. The submit page does not print those hours.

On 2026-11-03 the home page news said the list of accepted papers and the schedule were already on the NeurIPS portal, and that the submission deadline had been extended to September 2, 2026. Relative to this fetch, that deadline has passed. This note does not guess whether a late submission is still open.

## Required content

Full papers should contain, in the CFP’s words:

1. **Problem.** Biological task, data modality, setting, assumptions, target metrics, and the desired improvement.
2. **Proposed approach.** The method, its mechanism, preconditions, and the hypothesis under which it was expected to work.
3. **Observed outcome.** The negative, null, or unexpected outcome, with quantitative evidence where possible (metrics, uncertainty, compute, data, seeds).
4. **Reason for failure.** Why it did not work, with diagnostics or ablations, boundary conditions, and takeaways.

The CFP also asks for a short paragraph disclosing any use of large language models. It recommends ethics and reproducibility statements and says those do not count toward the page limit. The style file does not itself move those sections out of the page count; `main.tex` places them after the numbered sections and before the references, and the compile note records where each block starts.

Accepted papers appear on OpenReview. The workshop is non-archival. Work already published in conference proceedings is not eligible. Concurrent NeurIPS 2026 main-track review is allowed. The submit page says a checklist does not count toward the limit, but the downloaded zip has no checklist items. No checklist was invented for this draft.

## What was not confirmed

- Whether the NeurIPS portal still accepts papers after the accepted-list announcement.
- A checklist questionnaire. It is mentioned and not included in the style zip.
- The EPFL “December 11 or 12” wording. The workshop site says December 11, 2026.
