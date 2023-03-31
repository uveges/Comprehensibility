class WordFileTriplet(object):
    """
    Minden fájlnak 3 verziója van, pl:
        'A1A.txt.out', 'A1B.txt.out', 'A1C.txt.out'
    Az eleje azonos, az A jelzi az eredeti, a B a javított változatot (a C-ben ott van a korrektúra)

    Ezeket eltároljuk egy objektumban;
        - aminek a 'name' adattagja a közös rész a három fájlnévben,
        - a content_A, content_B és content_C pedig a megfelelő verzió szövegét tartalmazza, mondatokra szedve listában.
        - minden értéket csak egyszer állíthatunk be, ha már rendelkezik értékkel, akkor kivételt dobunk

    Metódusok:
        Setterek: name, content_A, content_B, content_C

    """

    def __init__(self, name=None, content_A=None, content_B=None, content_C=None):
        self._name = name
        self._content_A = content_A
        self._content_B = content_B
        self._content_C = content_C

    @property
    def name(self, ):
        return self._name

    @property
    def content_A(self):
        return self._content_A

    @property
    def content_B(self):
        return self._content_B

    @property
    def content_C(self):
        return self._content_C

    @name.setter
    def name(self, name):
        if self._name == None:
            self._name = name
        else:
            raise NameException

    @content_A.setter
    def content_A(self, content):
        if self._content_A is None:
            self._content_A = content
        else:
            raise ContentAexception

    @content_B.setter
    def content_B(self, content):
        if self._content_B is None:
            self._content_B = content
        else:
            raise ContentBexception

    @content_C.setter
    def content_C(self, content):
        if self._content_C is None:
            self._content_C = content
        else:
            raise ContentCexception

    def isAllset(self):
        if self._content_A is not None and self._content_B is not None and self._content_C is not None:
            return True
        return False

    def whichisempty(self):

        empty = list()

        if self._content_A is None:
            empty.append('A')
        if self._content_B is None:
            empty.append('B')
        if self._content_C is None:
            empty.append('C')

        return empty

    def get_subcorpora(self):

        """
        Visszaadja: (Lista, két belső listával)
        - az első listában azokat a mondatokat, amelyek csak az EREDETI VÁLTOZATBAN vannak meg
        - a másodikban pedig azokat, amelyek csak a KÖZÉRTHETŐ VÁLTOZATBAN
        :return:
        """
        if not self.isAllset():
            return ['Az objektumnak vannak beállítatlan elemei: '].append(str(self.whichisempty()))

        setA = set(self._content_A)
        setB = set(self._content_B)

        # A: eredeti változat
        # B: közérthető változat
        # C: korrektúrázott változat

        subcorpora = list()

        # elsőnek vissza join-oljuk a rosszul szegmentált mondatokat

        subcorpora.append(setA.difference(setB))
        subcorpora.append(setB.difference(setA))

        return subcorpora


class NameException(Exception):
    # print something to sys err?
    ...


class ContentAexception(Exception):
    ...


class ContentBexception(Exception):
    ...


class ContentCexception(Exception):
    ...
