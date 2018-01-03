-- Find users that are students and that are not affiliated with any teachers. 
-- Find all of their clickstream data

WITH select_users AS
(
    SELECT id
    FROM users
    WHERE role = 'STUDENT'
    EXCEPT
    SELECT user_id_granting_permission AS id
    FROM user_associations
        LEFT JOIN users
        ON users.id = user_id_granting_permission
)
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
