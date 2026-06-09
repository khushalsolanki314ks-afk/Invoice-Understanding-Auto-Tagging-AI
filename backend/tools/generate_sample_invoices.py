# Generates simple one-page invoice PDFs in multiple languages using ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from pathlib import Path

OUT = Path(__file__).parent.parent.parent / "data" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    'invoice_en.pdf': {
        'lang': 'English',
        'lines': [
            'INVOICE',
            'Invoice No: EN-2026-0001',
            'Invoice Date: 06/06/2026',
            'Due Date: 06/07/2026',
            '',
            'Seller: Apex Packaging Co.',
            'Buyer: Reliance Retail Ltd',
            '',
            'Description           Qty   Rate    Amount',
            'HDPE Bags 50kg         500  650.00  325,000',
            'Corrugated Boxes L     200  767.50  153,500',
            '',
            'Subtotal: 478,500',
            'GST @18%: 86,130',
            'TOTAL: 564,630',
        ]
    },
    'invoice_hi.pdf': {
        'lang': 'Hindi',
        'lines': [
            'चालान',
            'चालान संख्या: HI-2026-0001',
            'चालान की तिथि: 06/06/2026',
            'देय तिथि: 06/07/2026',
            '',
            'विक्रेता: एपेक्स पैकेजिंग को.',
            'खरीदार: रिलायंस रिटेल लिमिटेड',
            '',
            'विवरण              मात्रा  दर     राशि',
            'HDPE बैग 50kg        500   650.00  3,25,000',
            'मूँड बक्से L         200   767.50  1,53,500',
            '',
            'उप-योग: 4,78,500',
            'GST @18%: 86,130',
            'कुल: 5,64,630',
        ]
    },
    'invoice_fr.pdf': {
        'lang': 'French',
        'lines': [
            'FACTURE',
            'N° Facture: FR-2026-0001',
            'Date: 06/06/2026',
            'Échéance: 06/07/2026',
            '',
            'Vendeur: Apex Packaging Co.',
            'Acheteur: Reliance Retail Ltd',
            '',
            'Description           Qté   Prix    Montant',
            'Sacs HDPE 50kg         500  650.00  325 000',
            'Boîtes cartonnées L    200  767.50  153 500',
            '',
            'Sous-total: 478 500',
            'TVA @18%: 86 130',
            'TOTAL: 564 630',
        ]
    },
    'invoice_es.pdf': {
        'lang': 'Spanish',
        'lines': [
            'FACTURA',
            'Nº Factura: ES-2026-0001',
            'Fecha: 06/06/2026',
            'Vencimiento: 06/07/2026',
            '',
            'Vendedor: Apex Packaging Co.',
            'Comprador: Reliance Retail Ltd',
            '',
            'Descripción           Cant   Precio   Importe',
            'Bolsas HDPE 50kg       500   650.00   325.000',
            'Cajas corrugadas L     200   767.50   153.500',
            '',
            'Subtotal: 478.500',
            'IVA @18%: 86.130',
            'TOTAL: 564.630',
        ]
    }
}

# Register a UTF-8 capable font (DejaVu Sans comes with reportlab or can be supplied)
try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
    FONT_NAME = 'DejaVuSans'
except Exception:
    # fallback to Helvetica
    FONT_NAME = 'Helvetica'

for fname, sample in SAMPLES.items():
    path = OUT / fname
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setFont(FONT_NAME, 14)
    y = height - 60
    for line in sample['lines']:
        c.drawString(60, y, line)
        y -= 18
    c.showPage()
    c.save()

print(f"Generated {len(SAMPLES)} sample invoices in: {OUT}")
