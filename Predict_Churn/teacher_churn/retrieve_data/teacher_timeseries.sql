-- Find users that are teachers and find all of their clickstream data
WITH select_users AS
(
    SELECT id
    FROM users
    WHERE role = 'TEACHER' AND school_id NOTNULL
    EXCEPT
    SELECT staff_users.id
    FROM staff_users
),
---------------------------------------------------------------------------
	-- Find all teacher logged events
---------------------------------------------------------------------------
add_logged_event AS
(
 SELECT select_users.id AS user_id,
        (CASE
        WHEN event_type = 'VIEW_HINT' THEN 'view_hint'
        WHEN event_type = 'VIDEO_PLAY' THEN 'play_video'
        WHEN event_type = 'VIEW_CONCEPT' THEN 'view_concept'
        WHEN event_type = 'VIEW_ASSIGNMENT_PROGRESS' THEN 'view_assig_prog'

        -- Attempted questions
        WHEN level = '0' AND event_details :: JSON ->> 'correct' = 'false' THEN 'q_lvl_0'
        WHEN level = '1' AND event_details :: JSON ->> 'correct' = 'false' THEN 'q_lvl_1'
        WHEN level = '2' AND event_details :: JSON ->> 'correct' = 'false' THEN 'q_lvl_2'
        WHEN level = '3' AND event_details :: JSON ->> 'correct' = 'false' THEN 'q_lvl_3'
        WHEN level = '4' AND event_details :: JSON ->> 'correct' = 'false' THEN'q_lvl_4'
        WHEN level = '5' AND event_details :: JSON ->> 'correct' = 'false' THEN 'q_lvl_5'
        WHEN level = '6' AND event_details :: JSON ->> 'correct' = 'false' THEN 'q_lvl_6'

        -- Correct questions
        WHEN level = '0' AND event_details :: JSON ->> 'correct' = 'true' THEN 'q_lvl_0_cor'
        WHEN level = '1' AND event_details :: JSON ->> 'correct' = 'true' THEN 'q_lvl_1_cor'
        WHEN level = '2' AND event_details :: JSON ->> 'correct' = 'true' THEN 'q_lvl_2_cor'
        WHEN level = '3' AND event_details :: JSON ->> 'correct' = 'true' THEN 'q_lvl_3_cor'
        WHEN level = '4' AND event_details :: JSON ->> 'correct' = 'true' THEN'q_lvl_4_cor'
        WHEN level = '5' AND event_details :: JSON ->> 'correct' = 'true' THEN 'q_lvl_5_cor'
        WHEN level = '6' AND event_details :: JSON ->> 'correct' = 'true' THEN 'q_lvl_6_cor'

                    ELSE NULL END) AS event_type,
	    timestamp AS date_of_event
	FROM
	select_users
	INNER JOIN logged_events
	ON select_users.id = logged_events.user_id
	LEFT JOIN content_data_asi_db
	ON event_details :: JSON ->> 'questionId' = content_data_asi_db.question_id
	WHERE event_type = 'VIEW_HINT' or event_type = 'VIDEO_PLAY'
	    or event_type = 'ANSWER_QUESTION' or event_type = 'VIEW_CONCEPT'
	    OR event_type = 'VIEW_ASSIGNMENT_PROGRESS'
),
---------------------------------------------------------------------------
	-- Find all add_user events for each teacher
---------------------------------------------------------------------------
	add_user_event AS
	(
	SELECT user_associations.user_id_receiving_permission AS user_id,
    CAST('add_user' AS text) AS event_type,
    user_associations.created AS date_of_event
	FROM select_users
	RIGHT JOIN user_associations
	ON user_associations.user_id_receiving_permission = select_users.id
	),
---------------------------------------------------------------------------
-- Find all add_assignment events for each teacher
---------------------------------------------------------------------------
	add_assig_event AS
(
	SELECT select_users.id AS user_id,
	  CASE WHEN gameboards.owner_user_id = assignments.owner_user_id THEN CAST('add_assig' AS text) ELSE CAST('add_custom_assig' AS text) END AS event_type,
    assignments.creation_date AS date_of_event
	FROM select_users, assignments
	LEFT JOIN gameboards
	ON gameboards.id = assignments.gameboard_id
	WHERE select_users.id = assignments.owner_user_id
),
---------------------------------------------------------------------------
-- Find all create_group events
---------------------------------------------------------------------------
add_group_event AS (
	SELECT groups.owner_id AS user_id,
	CAST('create_group' AS text) AS event_type,
  created   AS date_of_event
	FROM
	select_users
	RIGHT JOIN groups
		ON groups.owner_id = select_users.id
)
SELECT *
FROM add_group_event
UNION ALL
SELECT *
FROM add_assig_event
UNION ALL
SELECT *
FROM add_user_event
UNION ALL
SELECT *
FROM add_logged_event
ORDER BY user_id, date_of_event
