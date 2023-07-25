
-- Creating the table "users"
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  age INT
);



-- Inserting data into the "users" table
INSERT INTO users (id, name, age)
VALUES
  (1, 'John Doe', 30),
  (2, 'Jane Smith', 25),
  (3, 'Bob Johnson', 40),
  (4, 'Alice Williams', 22);

--Roll up, cube , grouping sets operation on above table
SELECT name, age, COUNT(*) AS total_users, AVG(age) AS avg_age
FROM users
GROUP BY ROLLUP(name, age);

SELECT name, age, COUNT(*) AS total_users, AVG(age) AS avg_age
FROM users
GROUP BY CUBE(name, age);


SELECT name, age, COUNT(*) AS total_users, AVG(age) AS avg_age
FROM users
GROUP BY GROUPING SETS((name, age), (name), ());




-- OLAP stands for "Online Analytical Processing," which is a category of software tools used to analyze multidimensional data from different perspectives. OLAP systems are designed to efficiently and quickly perform complex analytical queries on large datasets.

-- ****OLAP queries in SQL— These SQL query constructs that are used for data aggregation and conditional calculations in OLAP cubes.****

-- 1-The **CUBE** operator computes a union of GROUP BY’s on every subset of the specified attribute types


SELECT QUARTER, REGION, SUM(SALES)
FROM SALESTABLE
GROUP BY CUBE (QUARTER, REGION)


  
CASH WHEN

```sql
CASE
    WHEN condition1 THEN value1
    WHEN condition2 THEN value2
    ...
    ELSE value
END



SELECT CASE WHEN grouping(QUARTER) = 1 THEN 'All' ELSE QUARTER END AS QUARTER, CASE WHEN grouping(REGION) = 1 THEN 'All' ELSE REGION END AS REGION, SUM(SALES)
FROM SALESTABLE
GROUP BY CUBE (QUARTER, REGION)


-- The **ROLLUP** operator computes the union on every prefix of the list of specified attribute types, from the most detailed up to the grand total.


SELECT QUARTER, REGION, SUM(SALES)
FROM SALESTABLE
GROUP BY ROLLUP (QUARTER, REGION)
  

-- The **GROUPING SETS** operator generates a result set equivalent to that generated by a UNION ALL of multiple simple GROUP BY clauses.


SELECT QUARTER, REGION, SUM(SALES)
FROM SALESTABLE
GROUP BY GROUPING SETS ((QUARTER), (REGION))


-- This query is equivalent to:

SELECT QUARTER, NULL, SUM(SALES)
FROM SALESTABLE
GROUP BY QUARTER
UNION ALL
SELECT NULL, REGION, SUM(SALES)
FROM SALESTABLE
GROUP BY REGION