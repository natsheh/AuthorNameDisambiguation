PAPER_ALL_INFO = """
SELECT 
v.*,
GROUP_CONCAT( DISTINCT COALESCE(ar.first_author, ' ') ORDER BY ar.first_author SEPARATOR ' ') AS ref_authors,
GROUP_CONCAT( DISTINCT COALESCE(ar.journal, ' ') ORDER BY ar.journal SEPARATOR ' ') AS ref_journals,
GROUP_CONCAT( DISTINCT COALESCE(ac.country, ' ') ORDER BY ac.country SEPARATOR ' ') AS country
FROM
(
    SELECT 
        t.author_id,
        t.author_name,
        t.article_id,
        t.article_title, 
        t.year,
        t.authors,
        t.subjects, 
        t.keywords,
        t.journal,
        t.institution
    FROM (
        SELECT
            aad.authorId as author_id,
            aad.author as author_name,
            a.id as article_id,
            a.title as article_title,
            a.year as year,
            a.abbr as journal, #a.journal as journal
            a.authors as authors,
            GROUP_CONCAT( DISTINCT COALESCE(asu.subject, ' ') ORDER BY asu.subject SEPARATOR ' ') AS subjects,
            GROUP_CONCAT( DISTINCT COALESCE(ak.keyword, ' ') ORDER BY ak.keyword SEPARATOR ' ') AS keywords,
            GROUP_CONCAT( DISTINCT COALESCE(p.institution, ' ') ORDER BY p.institution SEPARATOR ' ') AS institution
        FROM
            articles a 
            JOIN articles_authors_disambiguated aad ON aad.id = a.id 
            LEFT JOIN articles_subjects asu ON asu.id = a.id
            LEFT JOIN articles_keywords ak ON ak.id = a.id
            LEFT JOIN pub_inst p ON p.pub_id = a.id
        WHERE aad.author LIKE "{0}"
        GROUP BY article_id, author_id
        ORDER BY article_id
    ) t
    ORDER BY author_id
) v
LEFT JOIN articles_refs ar ON ar.id = v.article_id
LEFT JOIN articles_countries ac ON ac.id = v.article_id
GROUP BY v.article_id, v.author_id 
ORDER BY v.author_id
"""

FOCUS_NAMES = """
SELECT 
     LOWER(TRIM(aad.author)) as author, aad.id
FROM 
    `articles_authors_disambiguated` as aad
WHERE
    aad.`author` like '{0}'
"""

AUTHOR_ORDERING = """
SELECT aad.id
FROM `articles_authors_disambiguated`aad 
WHERE aad.`author` LIKE '{0}'
GROUP BY aad.id, aad.d
ORDER BY aad.authorId
"""