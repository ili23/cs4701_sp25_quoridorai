def argmax(lst, key=None):
    m = lst[0]
    i_m = 0

    for i, o in enumerate(lst):
        v = o if not key else key(o)
        if v > m:
            m = v
            m_i = i
    return m_i
