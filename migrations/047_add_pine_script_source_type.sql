-- Add pine_script to documents.source_type check constraint

ALTER TABLE documents DROP CONSTRAINT documents_source_type_check;

ALTER TABLE documents ADD CONSTRAINT documents_source_type_check 
CHECK (source_type = ANY (ARRAY['youtube'::text, 'pdf'::text, 'article'::text, 'note'::text, 'transcript'::text, 'pine_script'::text]));
