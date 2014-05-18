GRAPH_PREPROCESSING = """DROP TABLE IF EXISTS graph_mv_unfiltered_graph;
CREATE TABLE graph_mv_unfiltered_graph(
    node1 int,
    node2 int
);"""

GRAPH_POSTPROCESSING = """
ALTER TABLE `DMKM_publications`.`graph_mv_unfiltered_graph` ADD INDEX `ind_node2_noed1` ( `node2` , `node1` );

DROP TABLE IF EXISTS graph_mv_filtered_graph;
CREATE TABLE graph_mv_filtered_graph(
    node1 int,
    node2 int
);

INSERT INTO graph_mv_filtered_graph
Select node1, node2 
from (SELECT node2 as n2, count(distinct node1) as inn from graph_mv_unfiltered_graph group by node2 having inn>1) as tmp, 
     graph_mv_unfiltered_graph as g 
where g.node2 = n2
group by node1, node2
order by node1, node2;

ALTER TABLE `DMKM_publications`.`graph_mv_filtered_graph` ADD INDEX `idx` ( `node2` , `node1` );
ALTER TABLE `graph_mv_filtered_graph` ADD INDEX ( `node1` );

DROP TABLE IF EXISTS graph_mv_intersection_count;
CREATE TABLE graph_mv_intersection_count(
    node1 int,
    node2 int,
    cnt float
);
INSERT INTO graph_mv_intersection_count
SELECT g1.node1 AS node1, g2.node1 AS node2, COUNT( DISTINCT  g2.node2)  AS cnt
FROM graph_mv_filtered_graph AS g1, graph_mv_filtered_graph AS g2
WHERE g1.node2 = g2.node2
AND g1.node1 < g2.node1
GROUP BY g1.node1, g2.node1
ORDER BY cnt DESC;

ALTER TABLE `graph_mv_intersection_count` ADD INDEX ( `node1` );

SELECT 
node1 as n1, 
aad1.authorId as author1,
node2 as n2,
aad2.authorId as author2,
cnt/GREATEST(( SELECT COUNT( * ) FROM graph_mv_filtered_graph WHERE graph_mv_filtered_graph.node1 = n1),
            (SELECT COUNT( * ) FROM graph_mv_filtered_graph WHERE graph_mv_filtered_graph.node1 = n2)) as score
FROM graph_mv_intersection_count, articles_authors_disambiguated aad1, articles_authors_disambiguated aad2
WHERE aad1.id = node1
AND aad2.id = node2
AND aad1.authorId >0
AND aad2.authorId >0
AND aad1.author like "{0}" 
AND aad2.author like "{0}" 
GROUP BY node1, node2
ORDER BY `score` DESC;

"""

GRAPH_REFERENCES_QUERY = """
INSERT INTO graph_mv_unfiltered_graph
SELECT refs1.id as node1, refs2.id as node2
FROM articles_refs_clean_2 as refs1,  articles_refs_clean_2 as refs2, articles_authors_disambiguated as authors
WHERE
refs1.id != refs2.id and
authors.author like "{0}" AND  authors.id = refs1.id AND
refs1.first_author = refs2.first_author AND
refs1.year = refs2.year AND
refs1.journal = refs2.journal;
""" 

GRAPH_COAUTHORSHIP_QUERY = """
INSERT INTO graph_mv_unfiltered_graph
SELECT authors1.id as node1, authors3.id as node2
FROM articles_authors_disambiguated as authors1, articles_authors_disambiguated as authors2, articles_authors_disambiguated as authors3
WHERE 
authors1.author LIKE "{0}" AND
authors2.author != authors1.author AND
authors3.author != authors1.author AND
authors2.author = authors3.author AND
authors1.id = authors2.id AND
authors1.id != authors3.id;
"""

GRAPH_KEYWORDS_QUERY = """
INSERT INTO graph_mv_unfiltered_graph
SELECT kw1.id AS node1, kw2.id AS node2
FROM articles_authors_disambiguated AS authors1, articles_keywords_clean_sel AS kw1, articles_keywords_clean_sel AS kw2
WHERE authors1.author like "{0}"
AND kw1.id = authors1.id
AND kw2.id != kw1.id
AND kw1.keyword = kw2.keyword
GROUP BY kw1.id, kw2.id;
"""

GRAPH_SUBJECTS_QUERY = """
INSERT INTO graph_mv_unfiltered_graph
SELECT authors.id as node1, sa.id as node2
FROM subject_asociations as sa, articles_authors_disambiguated as authors
WHERE
authors.id != sa.id AND
authors.author like  "{0}" AND
(SELECT count(distinct subject) from articles_subjects as asj where asj.id = authors.id) > 1 AND
NOT EXISTS (Select * from articles_subjects as s where
s.id = authors.id AND
s.subject NOT IN (Select subject from subject_asociations as sa2 where sa2.id = sa.id) );

INSERT INTO graph_mv_unfiltered_graph
SELECT authors.id as node1, sa.id as node2
FROM subject_asociations as sa, articles_authors_disambiguated as authors,articles_subjects as s
WHERE
authors.id != sa.id AND
authors.author like "{0}" AND
(SELECT count(distinct subject) from articles_subjects as asj where asj.id = authors.id) = 1
AND sa.id in (select node2 from graph_mv_filtered_graph)
AND sa.subject = s.subject
AND s.id = authors.id
GROUP BY node1, node2;
"""

COAUTHORSHIP_GRAPH_MATRIX = GRAPH_PREPROCESSING + GRAPH_COAUTHORSHIP_QUERY + GRAPH_POSTPROCESSING
REFERENCES_GRAPH_MATRIX = GRAPH_PREPROCESSING + GRAPH_REFERENCES_QUERY + GRAPH_POSTPROCESSING
SUBJECTS_GRAPH_MATRIX = GRAPH_PREPROCESSING + GRAPH_SUBJECTS_QUERY + GRAPH_POSTPROCESSING
KEYWORDS_GRAPH_MATRIX = GRAPH_PREPROCESSING + GRAPH_KEYWORDS_QUERY + GRAPH_POSTPROCESSING