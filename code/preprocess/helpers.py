def text_to_labels(text):
    """
    convert text to numbers label
    """
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret

def labels_to_text(labels):
    """
    convert numbers label to text
    """
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text

def get_list_safe(l, index, size):
    ret = l[index : index + size]
    while size - len(ret) > 0:
        ret += l[0 : size - len(ret)]
    return ret