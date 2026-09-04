"""
Erzeugt einen Standortplan als downloadbares PDF (in-memory) - Zusammenfassung
+ Koordinatenliste der gewählten Fahrzeugstandorte.

Umlaute sind unproblematisch (Latin-1, von der FPDF-Kernschrift Helvetica
unterstützt) - vermieden werden nur echte Sonderzeichen wie Halbgeviertstriche
(–) und das Euro-Zeichen (€), die die Kernschrift nicht darstellen kann.
"""

import time


def generate_location_plan_pdf(label, chosen_indices, candidate_sites, metrics_naive, metrics_hqm, n_servers, utilization):
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Standortplan - {label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Erstellt: {time.strftime('%d.%m.%Y %H:%M')} Uhr", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Zusammenfassung", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Anzahl Fahrzeuge: {n_servers}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Ziel-Systemauslastung: {utilization*100:.0f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 6, "", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(60, 6, "Naiv (Coverage)", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(60, 6, "HQM-bewusst", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 6, "Reaktionszeit (real, über HQM)", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(60, 6, f"{metrics_naive['art_served']:.2f}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(60, 6, f"{metrics_hqm['art_served']:.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(60, 6, "Abdeckung", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(60, 6, f"{metrics_naive['coverage_pct']:.1f}%", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(60, 6, f"{metrics_hqm['coverage_pct']:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(60, 6, "Verlustwahrscheinlichkeit", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(60, 6, f"{metrics_naive['p_loss']*100:.1f}%", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(60, 6, f"{metrics_hqm['p_loss']*100:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Fahrzeugstandorte (HQM-bewusste Lösung)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    headers = ["#", "x-Position", "y-Position"]
    widths = [15, 60, 60]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(235, 235, 235)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(7)

    pdf.set_font("Helvetica", "", 9)
    for i, idx in enumerate(chosen_indices):
        x, y = candidate_sites[idx]
        row = [str(i + 1), f"{x:.2f}", f"{y:.2f}"]
        for val, w in zip(row, widths):
            pdf.cell(w, 6, val, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(6)

    return bytes(pdf.output())
