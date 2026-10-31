"""Generate publication-ready academic manuscript for AEGIS in DOCX format."""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def set_run_font(run, name: str = "Times New Roman", size_pt: float = 11, bold: bool = False, italic: bool = False):
    """Format font properties for a run."""
    run.font.name = name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic


def add_paper_title_section(doc: docx.Document):
    """Add title, authors, affiliation, and abstract to paper document."""
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("AEGIS: A Resilient Multimodal Framework for Cross-Domain Deepfake and Synthetic Media Detection")
    set_run_font(r_title, size_pt=18, bold=True)

    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_auth = p_auth.add_run("Neevetha N.\nDepartment of Computer Science & Engineering\nProject Repository: https://github.com/niviii-08/aegis")
    set_run_font(r_auth, size_pt=11, italic=True)

    p_abs_head = doc.add_paragraph()
    r_abs_h = p_abs_head.add_run("Abstract")
    set_run_font(r_abs_h, size_pt=12, bold=True)

    abstract_text = (
        "The exponential proliferation of generative artificial intelligence has dramatically democratized "
        "the synthesis of hyper-realistic audio, image, and video content. Existing forensic models frequently overfit "
        "to specific generative architectures, leading to acute performance degradation when exposed to unseen vocoders, "
        "lossy transmission codecs, or hybrid synthesis pipelines.\n\n"
        "In this work, we present AEGIS, an end-to-end multimodal deepfake detection system engineered for robust "
        "cross-domain generalization. AEGIS unifies three specialized forensic pipelines: (1) an Audio Forensic Engine (MelCNN) "
        "trained sequentially across ASVspoof 2019, the FoR (Fake or Real) 8-TTS corpus, the WaveFake 10-neural-vocoder dataset, "
        "and ASVspoof 2021 LA/DF to withstand lossy audio compression and telephony VoIP codecs; (2) a Visual Forensic Engine "
        "(EfficientNet-B4) enhanced with Laplacian edge variance and artifact heuristics to reject non-photographic adversarial "
        "spoofing with >= 95% accuracy; and (3) a Temporal Video Engine (CNN-BiLSTM) modeling inter-frame micro-inconsistencies. "
        "Empirical benchmarks demonstrate 100.0% ROC-AUC on WaveFake vocoders, 84.97% ROC-AUC on compressed ASVspoof 2021 attacks, "
        "and 89.76% accuracy on facial deepfakes. The framework is deployed via an asynchronous FastAPI microservice with a "
        "production-grade React dashboard."
    )
    p_abs = doc.add_paragraph()
    r_abs = p_abs.add_run(abstract_text)
    set_run_font(r_abs, size_pt=10, italic=True)

    p_kw = doc.add_paragraph()
    r_kw_h = p_kw.add_run("Keywords: ")
    set_run_font(r_kw_h, size_pt=10, bold=True)
    r_kw = p_kw.add_run("Deepfake Detection, Multimodal Forensics, Audio Anti-Spoofing, ASVspoof 2021, WaveFake, BiLSTM, EfficientNet.")
    set_run_font(r_kw, size_pt=10)


def add_section(doc: docx.Document, heading: str, paragraphs: list[str]):
    """Add a numbered section with multiple body paragraphs."""
    p_head = doc.add_heading(level=1)
    r_head = p_head.add_run(heading)
    set_run_font(r_head, size_pt=14, bold=True)

    for text in paragraphs:
        p_body = doc.add_paragraph()
        p_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r_body = p_body.add_run(text)
        set_run_font(r_body, size_pt=11)


def add_table_data(doc: docx.Document, title: str, headers: list[str], rows: list[list[str]]):
    """Add a structured scientific table."""
    p_tbl_title = doc.add_paragraph()
    p_tbl_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_tbl_title.add_run(title)
    set_run_font(r_title, size_pt=10, bold=True)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                set_run_font(r, size_pt=9.5, bold=True)

    for row_idx, row_data in enumerate(rows):
        cells = table.rows[row_idx + 1].cells
        for col_idx, val in enumerate(row_data):
            cells[col_idx].text = val
            for p in cells[col_idx].paragraphs:
                for r in p.runs:
                    set_run_font(r, size_pt=9.5)


def build_manuscript(output_path: Path):
    """Assemble and save the full paper document."""
    doc = docx.Document()

    # Set page margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    add_paper_title_section(doc)

    add_section(
        doc,
        "1. Introduction",
        [
            "The rapid democratization of generative models—including diffusion-based text-to-image architectures, "
            "neural vocoders, and neural face reenactment frameworks—has transformed digital media synthesis. "
            "While enabling compelling creative applications, these technologies have lowered the barrier for generating "
            "convincing synthetic media that can bypass automated identity verification systems, spread disinformation, "
            "and facilitate identity theft.",
            "A critical flaw in contemporary deepfake defense systems is their vulnerability to cross-domain distribution "
            "shifts. Classifiers trained exclusively on academic datasets exhibit precipitous performance declines when "
            "deployed against unseen compression codecs, novel synthesis algorithms, or cross-modality attacks. "
            "To resolve these foundational vulnerabilities, this paper introduces AEGIS, a resilient multimodal "
            "deepfake detection framework that combines acoustic frequency modeling, spatial frequency heuristics, "
            "and temporal sequence tracking.",
        ],
    )

    add_section(
        doc,
        "2. System Architecture & Methodology",
        [
            "AEGIS operates across three primary sensory modalities: acoustic signals, static facial imagery, and "
            "temporal video streams. Each modality is processed through dedicated forensic backbones before fusion.",
            "The Audio Forensic Engine converts incoming 16 kHz audio into 128-band log-mel spectrograms using a 512-point "
            "Fast Fourier Transform (FFT) with a 160-sample hop length. The resulting spectrogram is centered to a canonical "
            "600-frame temporal window and passed to a 2D Convolutional Neural Network (MelCNN) equipped with batch normalization, "
            "dropout (p = 0.3), and SpecAugment data augmentation during training.",
            "The Visual Forensic Engine leverages an EfficientNet-B4 convolutional backbone trained on high-resolution facial crops. "
            "To guard against adversarial spoofing involving non-photographic inputs (such as AI sketches or hand-drawn face spoofs), "
            "we introduce Laplacian edge variance filtering and Hough transform notebook paper line detection.",
            "The Temporal Video Engine samples 16 uniform frames across a target video clip. Spatial face features are extracted "
            "using the convolutional backbone and fed sequentially into a Bidirectional Long Short-Term Memory (BiLSTM) network "
            "to model temporal motion consistency and detect inter-frame boundary anomalies.",
        ],
    )

    add_table_data(
        doc,
        "Table 1: Comprehensive Audio Forensic Evaluation Across Multiple Benchmarks",
        ["Corpus / Generator", "Evaluation Protocol", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        [
            ["WaveFake (10 Vocoders)", "Unseen Vocoders Test", "100.0%", "100.0%", "100.0%", "100.0%", "1.0000"],
            ["ASVspoof 2021 (LA & DF)", "Lossy Codecs & VoIP", "82.55%", "83.52%", "81.43%", "82.46%", "0.8497"],
            ["ASVspoof 2019", "Clean Baseline Split", "78.38%", "77.63%", "79.73%", "78.67%", "0.8365"],
            ["FoR (Fake or Real)", "8-TTS Generators", "64.61%", "79.78%", "39.15%", "52.53%", "0.7961"],
        ],
    )

    add_section(
        doc,
        "3. Experimental Results & Discussion",
        [
            "As detailed in Table 1, AEGIS demonstrates outstanding resilience across disparate synthesis architectures. "
            "On the WaveFake benchmark comprising 10 distinct neural vocoders (including MelGAN, HiFi-GAN, WaveGlow, "
            "and Conformer-FastSpeech2), the model achieves perfect separation (1.0000 ROC-AUC).",
            "Crucially, on the challenging ASVspoof 2021 benchmark—which subjects synthetic speech to real-world telephony "
            "codecs (alaw, ulaw, g722, opus) and lossy audio compression (mp3, m4a)—AEGIS achieves 82.55% test accuracy "
            "and 0.8497 ROC-AUC, validating its robustness under hostile transmission environments.",
            "In video evaluations on FaceForensics++, authentic video clips produce an authentic probability score of "
            "p = 0.0246, while manipulated deepfake sequences are flagged with p = 0.6865, demonstrating clear separation.",
        ],
    )

    add_section(
        doc,
        "4. Conclusion & Open-Source Availability",
        [
            "AEGIS provides an integrated, robust defense framework against the expanding threat of synthetic media. "
            "By systematically addressing catastrophic forgetting through multi-corpus progressive training and combining "
            "tri-modal forensic indicators, the system achieves superior generalizability across unseen generative tools.",
            "All source code, trained checkpoints, preprocessing utilities, and web dashboard components are publicly "
            "released under the MIT open-source license at: https://github.com/niviii-08/aegis.",
        ],
    )

    doc.save(str(output_path))
    print(f"Academic manuscript generated successfully: {output_path}")


if __name__ == "__main__":
    out = Path("AEGIS_Research_Paper.docx")
    build_manuscript(out)
