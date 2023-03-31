import os

# Absolute paths required!
WORD_FILES_FOLDER = "/home/istvanu/DATA/KFO/"                   # should exist before run the program
MAIN_FOLDER_FOR_PROJECT = "/home/istvanu/DATA/"                 # should exist before run the program - used only for dump

# Set true, if any elements in FILTER dict. is True
FILTERING = True

# Specify type(s) of filtering
FILTER = {
    "headings_and_table_of_contents": True,
    "URLs": True,
    "minimum_characters_for_a_sentence": 30,                    # set to 0 if you don't want to filter sentences by char. number
    "punctuations": False
}

# .ending for temporary files
TEMP_FILE_EXTENSIONS = {
    "sentence_segmented": ".sent"
}

STOPWORDS = ["a", "abban", "ad", "ahhoz", "ahogy", "ahol", "aki", "akik", "akkor", "alá", "alatt", "által", "általában",
             "amely", "amelyek", "amelyekben", "amelyeket", "amelyet", "amelynek", "ami", "amíg", "amikor", "amit",
             "amolyan", "amúgy", "annak", "arra", "arról", "át", "az", "azért", "azok", "azon", "azonban", "azt",
             "aztán", "azután", "azzal", "bár", "be", "belül", "benne", "cikk", "cikkek", "cikkeket", "csak", "de", "e",
             "ebben", "eddig", "egész", "egy", "egyéb", "egyes", "egyetlen", "egyik", "egyre", "ehhez", "ekkor", "el",
             "elég", "ellen", "elő", "először", "előtt", "első", "emilyen", "én", "ennek", "éppen", "erre", "es", "és",
             "ez", "ezek", "ezen", "ezért", "ezt", "ezzel", "fel", "feladva", "felé", "főleg", "ha", "hanem", "hát",
             "hello", "helyett", "hiszen", "hogy", "hogyan", "hozzászólás", "hozzászólások", "ide", "ide", "igen",
             "így", "ill", "ill.", "illetve", "ilyen", "ilyenkor", "is", "ismét", "ison", "itt", "jó", "jobban",
             "jog fenntartva", "jól", "kategória", "kell", "kellett", "keressünk", "keresztül", "ki", "kívül",
             "komment", "között", "közül", "le", "legalább", "legyen", "lehet", "lehetett", "lenne", "lenni", "lesz",
             "lett", "maga", "magát", "majd", "már", "más", "másik", "meg", "még", "mellett", "mely", "melyek", "mert",
             "mi", "miért", "míg", "mikor", "milyen", "minden", "mindenki", "mindent", "mindig", "mint", "mintha",
             "mit", "mivel", "most", "nagy", "nagyobb", "nagyon", "ne", "néha", "néhány", "nekem", "neki", "nélkül",
             "nem", "nincs", "ő", "oda", "ők", "oka", "őket", "olyan", "ön", "os", "össze", "ott", "pedig", "persze",
             "rá", "s", "saját", "sem", "semmi", "soha", "sok", "sokat", "sokkal", "száma", "számára", "szemben",
             "szerint", "szerintem", "szerző", "szét", "szia", "szinte", "talán", "te", "tehát", "teljes", "ti", "több",
             "tovább", "továbbá", "üdv", "úgy", "ugyanis", "új", "újabb", "újra", "után", "utána", "utolsó", "vagy",
             "vagyis", "vagyok", "valaki", "valami", "valamint", "való", "van", "vannak", "vele", "vissza", "viszont",
             "volna", "volt", "volt", "voltak", "voltam", "voltunk"]