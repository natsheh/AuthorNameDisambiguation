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
            a.journal as journal,
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
        WHERE aad.authorId > 0 AND aad.author LIKE "{0}"
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

JARO_WINKLER_MASK = """
SELECT 
    aad1.id, aad1.authorId, aad2.id, aad2.authorId, 1 
FROM 
    `articles_authors_disambiguated` as aad1, `articles_authors_disambiguated` as aad2 
WHERE
    aad1.`author` like '{0}' and aad2.`author` LIKE '{0}' 
    AND jaro_winkler_similarity(SUBSTRING_INDEX(SUBSTRING_INDEX(aad1.author, ' ', 2), ' ', -1), SUBSTRING_INDEX(SUBSTRING_INDEX(aad2.author, ' ', 2), ' ', -1)) > 0
    AND aad1.authorId>0 and aad2.authorId>0
ORDER BY
    aad1.authorId
"""

FOCUS_NAMES = """
SELECT 
     LOWER(TRIM(aad.author)) as author, aad.id
FROM 
    `articles_authors_disambiguated` as aad
WHERE
    aad.`author` like '{0}'
    AND aad.authorId>0 
"""

AUTHOR_ORDERING = """
SELECT aad.id
FROM `articles_authors_disambiguated`aad 
WHERE aad.`author` LIKE '{0}' AND aad.authorId>0
GROUP BY aad.id, aad.d
ORDER BY aad.authorId
"""

BEST_GRAPH = """
SELECT node1 as n1, node2 as n2, cnt/GREATEST(( SELECT COUNT( * ) FROM graph_mv_filtered_graph WHERE graph_mv_filtered_graph.node1 = n1),
(SELECT COUNT( * ) FROM graph_mv_filtered_graph WHERE graph_mv_filtered_graph.node1 = n2)) as score, aad1.authorId as author1, aad2.authorId as author2
FROM graph_mv_intersection_count, articles_authors_disambiguated aad1, articles_authors_disambiguated aad2
WHERE aad1.id = node1
AND aad2.id = node2
AND aad1.authorId >0
AND aad2.authorId >0
GROUP BY node1, node2
ORDER BY `score` DESC
"""   

INSTITUTION_SQL = """
SELECT
    aad.authorId as author_id,
    aad.author as author_name,
    a.id as article_id,
    a.title as article_title,
    GROUP_CONCAT(ai.institution ORDER BY ai.institution SEPARATOR ' ')   AS institution
FROM
    articles a 
    JOIN articles_authors_disambiguated aad ON aad.id = a.id 
    LEFT JOIN articles_institutions ai ON ai.id = a.id 
WHERE aad.authorId > 0 AND aad.author LIKE "{0}"
GROUP BY ai.id, ai.d1
ORDER BY author_id
"""


COAUTHOR_SQL = """
SELECT
    aad.authorId as author_id,
    a.id as article_id,
    a.title as article_title,
    GROUP_CONCAT(aa.author ORDER BY aa.author SEPARATOR ' ')   AS coauthor
FROM
    articles a 
    JOIN articles_authors_disambiguated aad ON aad.id = a.id 
    LEFT JOIN articles_authors aa ON aa.id = a.id 
WHERE aad.authorId > 0 AND aad.author LIKE "{0}"
GROUP BY aa.id, aa.d
ORDER BY author_id
"""