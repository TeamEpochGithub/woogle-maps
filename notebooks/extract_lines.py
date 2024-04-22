# get the df from preprocessing.py
import pandas as pd

df = pd.read_csv('./data/selected/formatted_documents.csv', sep=',')
# df = df.filter(like='full_text_adj', axis='columns')
df.head()
# %%
df[['foi_fileName', 'full_text_adj']].style.set_table_styles([{
    'selector': '',
    'props': [('font-family', 'Arial, sans-serif'), ('border-collapse', 'collapse'), ('width', '100%')]
}, {
    'selector': 'td, th',
    'props': [('border', '1px solid #dddddd'), ('text-align', 'left'), ('padding', '8px')]
}, {
    'selector': 'tr:nth-child(even)',
    'props': [('background-color', '#f2f2f2')]
}, {
    'selector': 'th',
    'props': [('background-color', '#4CAF50'), ('color', 'white')]
}]).to_html('./data/selected/formatted_documents.html')
# %%
df[['foi_fileName', 'full_text_adj']].style.to_html('./data/selected/formatted_documents.html')
# %%
df = df.fillna('')


# %%%
# df['full_text_adj'] = df['full_text_adj'].str.split('\n')
# %%
# df['full_text_adj_filter'] = df['full_text_adj'].apply(lambda xs: [x for x in xs if len(x) > 30])
def paragraph_lines(lines: list[str], paragraphs):
    """
    Line above or below is a paragraph
    The line is more than k characters
    :param lines:
    :return:
    """
    paragraphs = [False if len(x) == 0 else None for x in lines]

    for line_idx, line in enumerate(lines):
        if paragraphs[line_idx] is not None:
            # already checked, e.g. at the beginning if empty
            return paragraphs[line_idx]
        # has text and the line below or above is a paragraph
        paragraphs[line_idx] = paragraph_lines(line_idx + 1, lines, paragraphs) or paragraph_lines(line_idx - 1, lines,
                                                                                               paragraphs)

def paragraph_lines_helper(lines: list[str]) -> list[bool]:
    paragraphs = [False if len(x) == 0 else None for x in lines]
    paragraph_lines(0, lines, paragraphs)


# %%
split_lines = df['full_text_adj'].str.split('\n').explode()
# %%
split_lines = split_lines.str.strip()
split_lines = split_lines[split_lines.str.len() > 0]
# groupby index and keep lines within each group if it is longer than 30 characters
k = 30
# %% identify paragraphs: two elements in the split_lines list which are longer than k characters

# df['split_lines'] = pd.Series(df.iloc[0]['full_text_adj'].split('\n'))
# %%
split_df = pd.DataFrame(split_lines)
split_df.rename(columns={0: 'split_lines'}, inplace=True)
# %%
split_df['split_lines'] = split_df['split_lines'].str.strip()
# %%
split_df['long_lines_30'] = split_df['split_lines'].apply(lambda x: x if len(x) > 30 else '')
# %% filter: keep rows which are longer than 20 characters
# or match the regex for a sentence: ^[A-Z].*[\.\!\?]$
sentence_regex = "\\s?[a-zA-Z]+[\\.,!?]"
char_limit_regex = ".{20,}"
long_lines = split_lines[split_lines.str.match(sentence_regex).fillna(False)]
# %%
long_lines = split_lines[split_lines.str.strip().str.len() > 20]
