import { createClient } from '@/lib/supabase/server';
import { pipeline } from '@xenova/transformers'
import { NextRequest } from 'next/server';

const generateEmbedding = await pipeline('feature-extraction', 'Supabase/gte-small');

interface Document {
    id: string;
    content: string;
    similarity: number;
}

async function SemanticSearch(embedding: number[], matchThreshold: number = 0.8, matchCount: number = 10): Promise<Document[] | undefined> {
    const supabase = await createClient();

    try {
        const { data: documents, error } = await supabase.rpc('semantic_query_match', {
            query_embedding: embedding,
            match_threshold: matchThreshold,
            match_count: matchCount
        });

        if (error) {
            console.error('Error fetching documents:', error);
            throw new Error('Failed to fetch matching documents');
        }
        return documents as Document[];
    } catch (error) {
        console.error("Error in fetchMatchingDocuments:", error);
        return undefined;
    }
}

export async function POST(req: NextRequest) {
    const { query } = await req.json();

    if (!query) {
        return new Response(JSON.stringify({ error: 'Query is required' }), { status: 400 });
    }

    const embedded_query = await generateEmbedding(
        query,
        {
            pooling: 'mean',
            normalize: true
        }
    );

    const actualQueryEmbedding = Array.from(embedded_query.data);

    console.log('Generated Query Embedding (first 10 elements):', actualQueryEmbedding.slice(0, 10), '...');
    console.log('Length of Query Embedding:', actualQueryEmbedding.length);

    const documents: Document[] | undefined = await SemanticSearch(actualQueryEmbedding);

    if (documents) {
        console.log('--- Matched Documents and their Similarity Scores ---');
        documents.forEach((doc: Document) => {
            console.log(`Document ID: ${doc.id}, Similarity: ${doc.similarity.toFixed(4)}, Content (excerpt): "${doc.content.substring(0, 100)}..."`);
        });
        return new Response(JSON.stringify({ documents }), { status: 200 });
    } else {
        return new Response(JSON.stringify({ error: 'No documents found or an error occurred during search' }), { status: 500 });
    }
}