import re
text = '''INVOICE

Invoice No: BM-2024-09001
Invoice Date: 01/09/2024
Due Date: 01/10/2024

Bill To:
Reliance Retail Ltd
Navi Mumbai, Maharashtra

From:
Apex Packaging Co.
45 Industrial Estate, Pune - 411001
'''
pat = re.compile(r"(?:invoice|inv|bill)\s*(?:no|number|#|num)?\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,})", re.IGNORECASE)
m = pat.search(text)
print('match:', bool(m))
if m:
    print('group1:', m.group(1))
else:
    print('no match')
