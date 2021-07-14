import os
import re
import argparse


# Paths used to save the datasets obtained
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_INTERIM_CORPUS = os.path.join(CURRENT_PATH, 'interim')

# Token used to identify the end of each post
END_OF_POST_TOKEN = '$END_OF_POST$'

# Regexes
UNICODE_REGEX = re.compile(r' #(?P<unicode>\d+);')
HTML_REGEX = re.compile(r'[ &](?P<html>amp|lt|gt);')
URL_FORMAT_PATTERN = re.compile(r'\[[^]]+?\]\(.+?\)')
WEB_URL_REGEX = re.compile(
    r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""")
SUB_REDDIT_REGEX = re.compile(r'/r/(?P<subreddit>[a-z0-9_]+?\b)')
NUMBER_REGEX = re.compile(r'\b[0-9]+?\b')
NOT_WORD_REGEX = re.compile(r"[^a-z0-9 ']")


def replace_unicode(match):
    """Replace unicode code value with symbol."""
    unicode_value = int(match.group('unicode'))
    return chr(unicode_value)


def replace_html_characters(match):
    """Replace HTML code value with symbol."""
    html_character = match.group('html')
    if html_character == 'amp':
        return '&'
    elif html_character == 'lt':
        return '<'
    elif html_character == 'gt':
        return '>'


def get_cleaned_post(post):
    """Clean post."""
    # Transform post to lowercase.
    clean_post = post.lower()
    # Replace unicode values with their symbol.
    clean_post = UNICODE_REGEX.sub(repl=replace_unicode, string=clean_post)
    # Replace the HTML codes with their symbol.
    clean_post = HTML_REGEX.sub(repl=replace_html_characters, string=clean_post)
    # Replace links to pages in reddit format with the token `weblink`.
    clean_post = URL_FORMAT_PATTERN.sub(repl='weblink', string=clean_post)
    # Replace direct links to pages with the token `weblink`.
    clean_post = WEB_URL_REGEX.sub(repl='weblink', string=clean_post)
    # Replace link to subreddit with the subreddit name.
    clean_post = SUB_REDDIT_REGEX.sub(repl=r'\g<subreddit>', string=clean_post)
    # Remove all characters except for letters, numbers and white spaces.
    clean_post = NOT_WORD_REGEX.sub(repl='', string=clean_post)
    # Replace numbers with the token `number`.
    clean_post = NUMBER_REGEX.sub(repl='number', string=clean_post)
    # Remove repeated white spaces, new lines and tabs.
    clean_post = " ".join(clean_post.split())
    # If the document ends up empty, add the word "empty" to represent it.
    clean_post = clean_post + 'empty' if clean_post == '' else clean_post
    return clean_post


def generate_clean_corpus(corpus_name, replace_old=True):
    """Pre-process the corpus.

    The pre-processing steps followed were:
        - convert text to lower case;
        - replace the decimal code for Unicode characters with its corresponding character;
        - replace HTML codes with their symbols;
        - replace reddit links to the web with the token weblink;
        - replace direct links to the web with the token weblink;
        - replace internal links to subreddits with the name of the subreddits;
        - delete any character that is not a number or letter;
        - replace numbers with the token number;
        - delete new lines, tab, and multiple consecutive white spaces.
    """
    interim_corpus_path = os.path.join(PATH_INTERIM_CORPUS, corpus_name)

    for stage in ['train', 'test']:
        input_corpus_path = os.path.join(interim_corpus_path, f'{corpus_name}-{stage}-raw.txt')
        output_file_name = f'{corpus_name}-{stage}-clean.txt'
        output_file_path = os.path.join(interim_corpus_path, output_file_name)
        print(f'Creating the corpus {output_file_name} ...')

        continue_processing_this_corpus = True

        if os.path.isfile(output_file_path):
            if replace_old:
                print(f'Cleaning the corpus {output_file_name} previously created')
                os.remove(output_file_path)
            else:
                print(f'The corpus {output_file_name} already exists. Delete it beforehand or '
                      f'call this function with the parameter `replace_old=True`.')
                continue_processing_this_corpus = False

        if continue_processing_this_corpus:
            with open(input_corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    label, document = line.split(maxsplit=1)
                    clean_document = ''
                    posts = document.split(END_OF_POST_TOKEN)
                    num_posts = len(posts)
                    for i, post in enumerate(posts):
                        clean_post = get_cleaned_post(post)
                        # Add the token of end of post
                        clean_post = clean_post + END_OF_POST_TOKEN if i < num_posts-1 else clean_post
                        clean_document = clean_document + clean_post
                    with open(output_file_path, 'a', encoding='utf-8') as f2:
                        f2.write(label + '\t' + clean_document + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to build a clean corpus.")
    parser.add_argument("corpus", help="eRisk task corpus name", choices=['t1', 't2'])
    args = parser.parse_args()

    generate_clean_corpus(corpus_name=args.corpus, replace_old=False)

    print('#' * 50)
    print('End of the script')
