// Friendly names for the generated contract types. Field names are the wire names.
import type { components } from './generated/openapi.js';

type S = components['schemas'];

export type { components, operations, paths } from './generated/openapi.js';

export type Tensor = S['Tensor'];
export type Base64Tensor = S['Base64Tensor'];
export type FloatTensor = S['FloatTensor'];
/** Explanation tensors: always float32. */
export type Float32Tensor = S['Float32Tensor'];
export type TensorDtype = S['TensorDtype'];
export type TensorAxis = S['TensorAxes'][number];
export type EncodingFormat = S['EncodingFormat'];
export type AttentionHeads = S['AttentionHeads'];
export type InputType = S['InputType'];
export type EncodeReturnField = S['EncodeReturnField'];

export type ModelRef = S['ModelRef'];
export type Representation = S['Representation'];
export type Model = S['Model'];
export type SearchProfile = S['SearchProfile'];
export type ScoringParameters = S['ScoringParameters'];
export type ServiceLimits = S['ServiceLimits'];
export type Usage = S['Usage'];
export type Token = S['Token'];
export type TextEncoding = S['TextEncoding'];
export type EncodedText = S['EncodedText'];
export type Explanation = S['Explanation'];
export type SlotVectors = S['SlotVectors'];

export type EncodeRequest = S['EncodeRequest'];
export type EncodeResponse = S['EncodeResponse'];
export type ScoreRequest = S['ScoreRequest'];
export type ScoreDocument = S['ScoreDocument'];
export type ScoreResponse = S['ScoreResponse'];
export type ScoreResult = S['ScoreResult'];

export type ChunkingConfig = S['ChunkingConfig'];
export type CreateIndexRequest = S['CreateIndexRequest'];
export type Index = S['Index'];
export type IndexStatus = S['IndexStatus'];
export type DeleteIndexResponse = S['DeleteIndexResponse'];
export type DocumentInput = S['DocumentInput'];
export type UpsertDocumentsRequest = S['UpsertDocumentsRequest'];
export type UpsertDocumentsResponse = S['UpsertDocumentsResponse'];
export type DocumentOutcome = S['DocumentOutcome'];
export type DocumentError = S['DocumentError'];
export type StoredDocument = S['StoredDocument'];
export type StoredChunk = S['StoredChunk'];
export type DeleteDocumentResponse = S['DeleteDocumentResponse'];
export type SearchRequest = S['SearchRequest'];
export type SearchResponse = S['SearchResponse'];
export type SearchResult = S['SearchResult'];

export type UsageSummary = S['UsageSummary'];
export type CallingApiKey = S['CallingApiKey'];
export type ApiKeyScope = S['ApiKeyScope'];
export type CounterString = S['CounterString'];
export type Metadata = S['Metadata'];
export type MetadataValue = S['MetadataValue'];
export type MetadataFilter = S['MetadataFilter'];
export type MetadataSchema = S['MetadataSchema'];

export type HealthStatus = S['HealthStatus'];
export type ReadinessStatus = S['ReadinessStatus'];
export type NotReadyStatus = S['NotReadyStatus'];
export type ErrorResponse = S['ErrorResponse'];
export type ErrorBody = S['Error'];
export type ErrorDetails = NonNullable<S['Error']['details']>;
export type IndexNotReadyDetails = S['IndexNotReadyDetails'];
export type ValidationIssue = S['ValidationIssue'];
