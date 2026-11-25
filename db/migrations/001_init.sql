CREATE TABLE IF NOT EXISTS epoch_counter (
  id int PRIMARY KEY CHECK (id = 1),
  epoch bigint NOT NULL
);
INSERT INTO epoch_counter (id, epoch) VALUES (1, 0) ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS documents (
  doc_id text,
  version int,
  tenant text,
  text text,
  born_epoch bigint NOT NULL,
  dead_epoch bigint NOT NULL DEFAULT 9223372036854775807,
  PRIMARY KEY (doc_id, version)
);

CREATE TABLE IF NOT EXISTS wal (
  epoch bigint,
  seq int,
  op text CHECK (op IN ('upsert','delete','acl_grant','acl_revoke')),
  doc_id text,
  version int,
  payload jsonb,
  PRIMARY KEY (epoch, seq)
);

CREATE TABLE IF NOT EXISTS acl_entries (
  doc_id text,
  principal text,
  granted_epoch bigint NOT NULL,
  revoked_epoch bigint NOT NULL DEFAULT 9223372036854775807
);

CREATE TABLE IF NOT EXISTS group_members (
  grp text,
  member text,
  added_epoch bigint,
  removed_epoch bigint DEFAULT 9223372036854775807
);
