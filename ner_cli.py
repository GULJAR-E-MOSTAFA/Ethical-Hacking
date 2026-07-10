#!/usr/bin/env python3
"""
Simple NER CLI using spaCy.

Usage examples:
  python ner_cli.py --text "Barack Obama was born in Hawaii." --format json
  python ner_cli.py --file article.txt --format plain
"""
import sys
import json
import click
import spacy

@click.command()
@click.option('--text', '-t', help='Text to analyze.')
@click.option('--file', '-f', 'file_', type=click.Path(exists=True), help='Path to text file to analyze.')
@click.option('--model', '-m', default='en_core_web_sm', help='spaCy model to use (default: en_core_web_sm).')
@click.option('--format', '-o', 'outfmt', type=click.Choice(['json','iob','plain']), default='json', help='Output format.')
def main(text, file_, model, outfmt):
    if not text and not file_:
        click.echo('Error: Provide --text or --file', err=True)
        sys.exit(1)

    if file_:
        with open(file_, 'r', encoding='utf8') as fh:
            text = fh.read()

    try:
        nlp = spacy.load(model)
    except Exception as e:
        click.echo(f'Failed to load model "{model}": {e}', err=True)
        click.echo('Install a model with: python -m spacy download en_core_web_sm', err=True)
        sys.exit(2)

    doc = nlp(text)
    ents = [
        {
            'text': ent.text,
            'label': ent.label_,
            'start_char': ent.start_char,
            'end_char': ent.end_char
        } for ent in doc.ents
    ]

    if outfmt == 'json':
        print(json.dumps({'text': text, 'entities': ents}, ensure_ascii=False, indent=2))
    elif outfmt == 'iob':
        for token in doc:
            iob = token.ent_iob_
            ent_type = token.ent_type_ if token.ent_type_ else ''
            print(f'{token.text}\t{iob}\t{ent_type}')
    else:  # plain
        if not doc.ents:
            print('No entities found.')
            return
        for ent in doc.ents:
            print(f'{ent.text} ({ent.label_})')

if __name__ == '__main__':
    main()
