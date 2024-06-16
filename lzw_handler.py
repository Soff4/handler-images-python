class LZW:
    def compress(self, uncompressed):
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        w = ""
        result = []
        for c in uncompressed:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                # Add wc to the dictionary.
                dictionary[wc] = dict_size
                dict_size += 1
                w = c

        if w:
            result.append(dictionary[w])
        return result

    def decompress(self, compressed):
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}

        compressed = iter(compressed)
        w = chr(next(compressed))
        result = [w]
        for k in compressed:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[0]
            else:
                raise ValueError('Bad compressed k: %s' % k)
            result.append(entry)

            dictionary[dict_size] = w + entry[0]
            dict_size += 1

            w = entry
        return ''.join(result)
