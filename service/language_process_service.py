class LanguageProcessService:

    def extract_from_file(self, file):
        data = []
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                data.append(line)
        return data
