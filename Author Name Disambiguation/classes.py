class Paper:
    id = None
    title = None
    author_id = None #the disambiguated one for test purpose
    author_name = None
    coauthors = None
    institutions = None
    journals = None
    countries = None
    year = None
    subjects = None
    keywords = None
    ref_authors = None
    ref_journals = None
    unique_identifier = None #paper_id concat author_id
    
    def __init__(self, paper_id, paper_title, author_id, author_name, 
        coauthors=None, institutions=None, journals=None, year=None, subjects=None, 
        keywords=None, ref_authors=None, ref_journals=None, countries=None):
        self.id = paper_id
        self.title = paper_title
        self.author_id = author_id
        self.author_name = author_name
        self.coauthors = coauthors
        self.institutions = institutions
        self.journals = journals
        self.year = year
        self.subjects = subjects
        self.keywords = keywords
        self.ref_authors = ref_authors
        self.ref_journals = ref_journals
        self.countries = countries
        self.unique_identifier = str(paper_id) + str(author_id)

