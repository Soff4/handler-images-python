class LZW:
    # Цей метод приймає рядок uncompressed, який потрібно стиснути.
    def compress(self, uncompressed):
        # Створюється словник dictionary, який містить 256 символів (від 0 до 255) як ключі, а їхні ASCII-коди як значення.
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        # Ініціалізуються порожня строка w і порожній список result.
        w = ""
        result = []
        # Це головний цикл, який проходить по кожному символу c у uncompressed. Він будує рядок wc, додаючи c до w. 
        # Якщо wc наявний у словнику, то він замінює w на wc. 
        # Інакше, він додає поточне значення w до result, додає wc до словника з новим індексом dict_size, збільшує dict_size і замінює w на c.
        for c in uncompressed:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c
        # Після циклу, якщо w не порожній, то значення w також додається до result.
        if w:
            result.append(dictionary[w])
        # Функція повертає стиснутий результат як список індексів словника.
        return result

    # Цей метод приймає стиснутий список індексів словника compressed і відновлює початковий рядок.
    def decompress(self, compressed):
        # Створюється словник dictionary, який містить 256 символів (від 0 до 255) як значення, а їхні ASCII-коди як ключі.
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
        # Перетворюємо compressed на ітератор, отримуємо перший елемент та ініціалізуємо w і result цим елементом.
        compressed = iter(compressed)
        w = chr(next(compressed))
        result = [w]
        # Це головний цикл, який проходить по кожному індексу k у compressed. 
        # Якщо k наявний у словнику, то його значення (entry) додається до result. 
        # Якщо k дорівнює dict_size, то entry будується з w та першого символу w. Інакше, генерується помилка. Д
        # алі, до словника додається нове значення w + entry[0] з індексом dict_size, dict_size збільшується, а w замінюється на entry.
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
        # Функція повертає відновлений рядок, який отриманий з'єднанням елементів у result.
        return ''.join(result)
