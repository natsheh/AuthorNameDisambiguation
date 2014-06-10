class Paper:
    id = None
    title = None
    author_id = None            #the disambiguated one for test purpose
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
    unique_identifier = None    #paper_id concat author_id
    printout = None             #printout for production environment
    
    def __init__(self, paper_id, paper_title, author_id, author_name, 
        coauthors=None, institutions=None, journals=None, year=None, subjects=None, 
        keywords=None, ref_authors=None, ref_journals=None, countries=None, 
        unique_identifier=None, printout=None):
        self.id = paper_id
        self.title = str(paper_title)
        self.author_id = author_id
        self.author_name = str(author_name)
        self.coauthors = str(coauthors)
        self.institutions = str(institutions)
        self.journals = str(journals)
        self.year = str(year)
        self.subjects = str(subjects)
        self.keywords = str(keywords)
        self.ref_authors = str(ref_authors)
        self.ref_journals = str(ref_journals)
        self.countries = str(countries)
        self.unique_identifier = str(paper_id) + str(author_id)
        self.printout = printout

