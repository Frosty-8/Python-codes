words = [
    ('work','working'),
    ('happy','happier'),
    ('Love','Lovely'),
    ('create','creative'),
    ('clear','clearing'),
]
print('Original   Transformed    Added   Delete')
print('----------------------------------------')

for orig,trans in words:
    add = ''
    delete = ''
    if trans.startswith(orig):
        add = trans[len(orig):]
    else:
        i=0
        while i < min(len(orig),len(trans)) and orig[i]==trans[i]:
            i+=1
        delete = orig[i:]
        add = trans[i:]
    print(f'{orig:<12}{trans:<15}{add:<7}{delete}')
