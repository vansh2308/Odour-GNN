

def label_encoding(x, domain:list):
    return [len(domain)-1] if x not in domain else [domain.index(x) ]

if __name__ == '__main__':
    pass