import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import pandas as pd
from dotenv import load_dotenv
import os
import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.rule import Rule
from rich import print as rprint
from fpdf import FPDF
from fpdf.enums import XPos, YPos
# Load the .env file from the parent directory
load_dotenv(dotenv_path="../.env")

from src.graph import build_graph

console = Console()

# =============================================================================
#  PDF REPORT CLASS
# =============================================================================
class ClinicalPDF(FPDF):
    def header(self):
        # Hospital name banner
        self.set_fill_color(0, 51, 102)  # Dark navy blue
        self.rect(0, 0, 210, 22, 'F')
        self.set_font("Helvetica", 'B', 14)
        self.set_text_color(255, 255, 255)
        self.set_y(6)
        self.cell(0, 10, "MediSys AI  |  Clinical Discharge Report", align='C',
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10,
                  f"CONFIDENTIAL - AI-Generated Clinical Aid | Page {self.page_no()} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  align='C')

    def section_title(self, title, r=0, g=51, b=102):
        self.ln(4)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", 'B', 11)
        self.cell(0, 8, f"  {title}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, text, line_height=6):
        self.set_font("Helvetica", size=10)
        self.set_text_color(40, 40, 40)
        # Reset to left margin first to guarantee full epw is available
        self.set_x(self.l_margin)
        clean = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(self.epw, line_height, clean)
        self.ln(2)

    def key_value(self, key, value, val_color=(40, 40, 40)):
        key_w = 55
        val_w = self.epw - key_w
        self.set_font("Helvetica", 'B', 10)
        self.set_text_color(60, 60, 60)
        self.cell(key_w, 7, f"{key}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", size=10)
        self.set_text_color(*val_color)
        self.multi_cell(val_w, 7, str(value))
        self.set_text_color(0, 0, 0)

    def bullet_item(self, text, color=(180, 0, 0)):
        bullet_w = 6
        val_w = self.epw - bullet_w
        self.set_font("Helvetica", size=10)
        self.set_text_color(*color)
        self.cell(bullet_w, 6, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_text_color(40, 40, 40)
        clean = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(val_w, 6, clean)


# Sanitize any Unicode chars that Helvetica (Latin-1) cannot handle
def sanitize(text: str) -> str:
    replacements = {
        "\u2014": "-",   # em dash —
        "\u2013": "-",   # en dash –
        "\u2012": "-",   # figure dash
        "\u2010": "-",   # hyphen
        "\u2192": "->",  # rightward arrow →
        "\u2190": "<-",  # leftward arrow ←
        "\u2022": "-",   # bullet •
        "\u2026": "...", # ellipsis …
        "\u2018": "'",   # left single quote '
        "\u2019": "'",   # right single quote '
        "\u201c": '"',   # left double quote "
        "\u201d": '"',   # right double quote "
        "\u00b0": " degrees",  # degree °
        "\u00b1": "+/-",        # plus-minus ±
        "\u00d7": "x",          # multiplication ×
        "\u00f7": "/",          # division ÷
        "\u2665": "<3",         # heart ♥
        "\u2713": "OK",         # checkmark ✓
        "\u2717": "X",          # cross ✗
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final fallback: remove anything still outside Latin-1 range
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf(subject_id, hadm_id, draft_summary, safety_approved,
                 safety_feedback, flags, patient_meds,
                 output_path="output/discharge_summary.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now = datetime.datetime.now()

    # Sanitize all LLM-generated text upfront
    draft_summary   = sanitize(draft_summary)
    safety_feedback = sanitize(safety_feedback)
    flags           = [sanitize(f) for f in flags]
    patient_meds    = [sanitize(m) for m in patient_meds]

    pdf = ClinicalPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Patient Info Block ──────────────────────────────────────────────────
    pdf.section_title("Patient & Admission Details", 0, 51, 102)
    pdf.key_value("Subject ID", subject_id)
    pdf.key_value("Admission ID (HADM)", hadm_id)
    pdf.key_value("Report Generated", now.strftime("%B %d, %Y at %I:%M %p"))
    pdf.key_value("Pipeline", "LangGraph Multi-Agent AI (NER -> Reconciler -> Drafter -> Safety)")

    # ── Ground Truth Meds ───────────────────────────────────────────────────
    pdf.section_title("Admitted Medication List (Ground Truth)", 30, 90, 30)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(30, 30, 30)
    # Print in 2-column style
    meds_clean = [m.encode('latin-1', 'replace').decode('latin-1') for m in patient_meds]
    half = (len(meds_clean) + 1) // 2
    col_w = 93
    x_start = pdf.get_x()
    y_start = pdf.get_y()
    for i, med in enumerate(meds_clean):
        if i == half:
            pdf.set_xy(x_start + col_w + 4, y_start)
        pdf.set_x(x_start if i < half else x_start + col_w + 4)
        pdf.cell(col_w, 5.5, f"  {i+1}. {med}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── SOAP Draft ─────────────────────────────────────────────────────────
    pdf.section_title("AI-Generated SOAP Discharge Note", 0, 51, 102)
    # Parse and render SOAP sections individually for better styling
    sections = {"Subjective": "", "Objective": "", "Assessment": "", "Plan": ""}
    current = None
    for line in draft_summary.splitlines():
        stripped = line.strip().lstrip("*# ").rstrip("*# :")
        if stripped in sections:
            current = stripped
        elif current:
            sections[current] += line + "\n"

    if any(sections.values()):
        section_colors = {
            "Subjective":  (0, 90, 160),
            "Objective":   (0, 120, 60),
            "Assessment":  (140, 70, 0),
            "Plan":        (100, 0, 120),
        }
        for sec, content in sections.items():
            if content.strip():
                r, g, b = section_colors[sec]
                pdf.set_font("Helvetica", 'B', 10)
                pdf.set_text_color(r, g, b)
                pdf.cell(0, 7, sec.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(40, 40, 40)
                pdf.body_text(content.strip())
    else:
        pdf.body_text(draft_summary)

    # ── Reconciliation Flags ────────────────────────────────────────────────
    if flags:
        pdf.section_title("Medication Reconciliation Flags", 150, 40, 0)
        for flag in flags:
            pdf.bullet_item(flag, color=(160, 40, 0))
        pdf.ln(2)

    # ── Safety Assessment ───────────────────────────────────────────────────
    status_color = (0, 120, 60) if safety_approved else (180, 0, 0)
    status_label = "APPROVED" if safety_approved else "NOT APPROVED - REQUIRES PHYSICIAN REVIEW"
    pdf.section_title("AI Safety Assessment", *status_color)
    pdf.key_value("Safety Status", status_label, val_color=status_color)
    pdf.key_value("Safety Feedback", "")
    pdf.body_text(safety_feedback)

    # ── Disclaimer ──────────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_fill_color(255, 248, 220)
    pdf.set_draw_color(200, 160, 0)
    pdf.set_font("Helvetica", 'I', 9)
    pdf.set_text_color(100, 70, 0)
    disclaimer = ("DISCLAIMER: This document was generated by an AI system and is intended solely as a "
                  "clinical decision-support aid. It must be reviewed, verified, and approved by a licensed "
                  "physician before use. Do not use as the sole basis for patient care decisions.")
    pdf.multi_cell(0, 6, disclaimer, border=1, fill=True)

    pdf.output(output_path)
    console.print(f"\n[bold green]✔  PDF Report saved → [underline]{output_path}[/underline][/bold green]")


# =============================================================================
#  MAIN PIPELINE
# =============================================================================
def test_pipeline():
    console.print(Rule("[bold blue]Clinical Discharge Summary Generator[/bold blue]"))

    console.print("[dim]Loading CSV data...[/dim]")
    try:
        notes_df         = pd.read_csv("data/NOTEEVENTS.csv")
        prescriptions_df = pd.read_csv("data/PRESCRIPTIONS.csv")
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] Could not find CSV files in data/. Run generate_synthetic_notes.py first.")
        return

    if len(notes_df) == 0:
        console.print("[red]No notes available.[/red]")
        return

    # Grab the first patient record
    test_note  = notes_df.iloc[0]
    subject_id = test_note['SUBJECT_ID']
    hadm_id    = test_note['HADM_ID']
    raw_text   = test_note['TEXT']
    patient_meds = prescriptions_df[prescriptions_df['hadm_id'] == hadm_id]['drug'].tolist()

    # Patient info panel
    meds_preview = ", ".join(patient_meds[:6]) + f" ... (+{len(patient_meds)-6} more)"
    console.print(Panel(
        f"[bold]Subject ID:[/bold]  [cyan]{subject_id}[/cyan]\n"
        f"[bold]Admission ID:[/bold] [cyan]{hadm_id}[/cyan]\n"
        f"[bold]Total Meds:[/bold]  [yellow]{len(patient_meds)}[/yellow]\n"
        f"[bold]Preview:[/bold]     [dim]{meds_preview}[/dim]",
        title="[bold blue]Patient Record Loaded[/bold blue]",
        border_style="blue"
    ))

    app = build_graph()
    inputs = {"raw_clinical_note": raw_text, "ground_truth_meds": patient_meds}

    console.print(Rule("[bold yellow]Executing LangGraph Multi-Agent Pipeline[/bold yellow]"))

    # Agent metadata for live display
    agent_meta = {
        "NER_Agent":              ("[cyan]",    "1/4", "Medical NER",        "Extracting entities from clinical notes..."),
        "Reconciler_Agent":      ("[magenta]", "2/4", "Reconciler",         "Comparing meds against ground truth..."),
        "Drafting_Agent":        ("[yellow]",  "3/4", "SOAP Drafter",       "Generating discharge summary..."),
        "Safety_Agent":          ("[red]",     "4/4", "Safety Judge",       "Running safety & hallucination check..."),
    }

    import time
    final_state = {}
    for step in app.stream(inputs):
        for node_name, node_output in step.items():
            color, num, label, desc = agent_meta.get(node_name, ("[white]", "?", node_name, ""))
            console.print(
                f"{color}  [{num}] {label:<20}[/]  "
                f"[bold green]DONE[/bold green]  "
                f"[dim]{desc}[/dim]"
            )
            final_state.update(node_output)

    console.print("[bold green]  >> All agents finished successfully![/bold green]")

    draft_summary   = final_state.get("draft_summary", "")
    safety_approved = final_state.get("safety_approved")
    safety_feedback = final_state.get("safety_feedback", "")
    flags           = final_state.get("reconciliation_flags", [])

    # ── Rich Terminal Output ─────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Results[/bold cyan]"))
    console.print(Panel(draft_summary, title="[bold cyan]FINAL DRAFT SUMMARY[/bold cyan]", border_style="cyan"))

    # Safety table
    table = Table(title="Safety & Reconciliation Report", show_header=True, header_style="bold white on dark_red" if not safety_approved else "bold white on dark_green")
    table.add_column("Field", style="bold", width=25)
    table.add_column("Value")
    table.add_row("Safety Approved", f"[green]YES[/green]" if safety_approved else "[red]NO — Requires Review[/red]")
    table.add_row("Safety Feedback", safety_feedback)
    console.print(table)

    if flags:
        flag_panel = "\n".join(f"[red]•[/red] {f}" for f in flags)
        console.print(Panel(flag_panel, title="[bold red]Reconciliation Flags[/bold red]", border_style="red"))

    # ── Generate PDF ─────────────────────────────────────────────────────────
    console.print(Rule("[bold green]Generating PDF Report[/bold green]"))
    generate_pdf(subject_id, hadm_id, draft_summary, safety_approved,
                 safety_feedback, flags, patient_meds)


if __name__ == "__main__":
    test_pipeline()

# How to run:
# py -m pip install -r requirements.txt
# py generate_synthetic_notes.py
# py -m src.main
