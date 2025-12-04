"""Initial database schema

Revision ID: 000000000001
Revises: 
Create Date: 2025-12-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import datetime


# revision identifiers, used by Alembic.
revision: str = '000000000001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('doc_id', sa.String(length=255), nullable=False),
        sa.Column('filename', sa.String(length=500), nullable=False),
        sa.Column('status', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('entities', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('doc_id')
    )
    # Create an index on doc_id for faster lookups
    op.create_index('idx_doc_id', 'documents', ['doc_id'])


def downgrade() -> None:
    # Drop the index first
    op.drop_index('idx_doc_id', table_name='documents')
    # Drop the documents table
    op.drop_table('documents')