from pathlib import Path
from pypdf import PdfReader
from src.legal_parser import extract_legal_blocks
from src.legal_chunker import chunk_legal_blocks

ROOT = Path('data/documents')
PDFS = [
    '1959uut001.pdf', '1959uut003.pdf', '1959uut004.pdf', '1959uut005.pdf',
    '2024UU005.pdf', '2024uu001.pdf', '2024uu002.pdf', '2024uu003.pdf',
    '2025perpres004.pdf', 'UUD1945.pdf'
]


def load(path):
    return [p.extract_text() or '' for p in PdfReader(str(path)).pages]


def main():
    print('='*80)
    print('LEGAL CHUNKER V2 REGRESSION TEST')
    print('='*80)
    total_merges = 0
    for name in PDFS:
        pages = load(ROOT/name)
        blocks = extract_legal_blocks(pages)
        chunks = chunk_legal_blocks(blocks)
        merges = sum(c.get('merged_blocks', 1)-1 for c in chunks)
        cross = [c for c in chunks if c.get('is_cross_page')]
        total_merges += merges
        print(f'{name:20} pages={len(pages):2} blocks={len(blocks):3} chunks={len(chunks):3} merges={merges:2}')
        for c in cross:
            print(
                f'  CROSS: Pasal {c.get("pasal")} ayat {c.get("ayat")} pages {c.get("source_pages")}')

    pages = load(ROOT/'2024uu001.pdf')
    chunks = chunk_legal_blocks(extract_legal_blocks(pages))
    target = [c for c in chunks if c.get(
        'pasal') == '17' and c.get('ayat') == '2a']
    print('\nTARGET 2024uu001 Pasal 17 Ayat 2a')
    print('count:', len(target))
    if target:
        c = target[0]
        print('pages:', c.get('source_pages'))
        print('merged_blocks:', c.get('merged_blocks'))
        print('is_cross_page:', c.get('is_cross_page'))
        print('contains continuation:',
              'Transaksi Elektronik yang memiliki risiko tinggi' in c.get('content', ''))

    print('\nTOTAL MERGES:', total_merges)


if __name__ == '__main__':
    main()
