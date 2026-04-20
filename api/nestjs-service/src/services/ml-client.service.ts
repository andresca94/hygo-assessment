import {
  BadGatewayException,
  GatewayTimeoutException,
  Injectable,
  ServiceUnavailableException,
} from '@nestjs/common';
import axios, { AxiosInstance } from 'axios';
import FormData from 'form-data';

import { CheckRequestDto } from '../dto/check-request.dto';
import { AgeSafetyBatchResult, AgeSafetyHealth, AgeSafetyResult } from '../types/age-safety.types';

@Injectable()
export class MlClientService {
  private readonly client: AxiosInstance;
  private readonly timeoutMs: number;
  private readonly batchTimeoutMs: number;

  constructor() {
    this.timeoutMs = Number(process.env.ML_INFERENCE_TIMEOUT_MS ?? 300000);
    this.batchTimeoutMs = Number(process.env.ML_INFERENCE_BATCH_TIMEOUT_MS ?? 900000);
    this.client = axios.create({
      baseURL: process.env.ML_INFERENCE_URL ?? 'http://localhost:8000',
      timeout: this.timeoutMs,
    });
  }

  private throwUpstreamError(error: unknown, action: string): never {
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNABORTED') {
        throw new GatewayTimeoutException(`${action} timed out while waiting for the inference service.`);
      }
      if (error.response) {
        throw new BadGatewayException(
          `${action} failed because the inference service returned ${error.response.status}.`,
        );
      }
    }
    throw new ServiceUnavailableException('Inference service is not reachable.');
  }

  async health(): Promise<AgeSafetyHealth> {
    try {
      const response = await this.client.get<AgeSafetyHealth>('/health');
      return response.data;
    } catch (error) {
      throw new ServiceUnavailableException('Inference service is not reachable.');
    }
  }

  async checkImage(file: Express.Multer.File, metadata: CheckRequestDto): Promise<AgeSafetyResult> {
    const form = new FormData();
    form.append('file', file.buffer, {
      contentType: file.mimetype,
      filename: file.originalname,
    });
    if (metadata.requestId) {
      form.append('requestId', metadata.requestId);
    }
    if (metadata.source) {
      form.append('source', metadata.source);
    }

    try {
      const response = await this.client.post<AgeSafetyResult>('/v1/age-safety/check', form, {
        headers: form.getHeaders(),
        maxBodyLength: Infinity,
        timeout: this.timeoutMs,
      });
      return response.data;
    } catch (error) {
      this.throwUpstreamError(error, 'Single-image check');
    }
  }

  async checkBatch(files: Express.Multer.File[], metadata: CheckRequestDto): Promise<AgeSafetyBatchResult> {
    const form = new FormData();
    for (const file of files) {
      form.append('files', file.buffer, {
        contentType: file.mimetype,
        filename: file.originalname,
      });
    }
    if (metadata.requestId) {
      form.append('requestId', metadata.requestId);
    }
    if (metadata.source) {
      form.append('source', metadata.source);
    }

    try {
      const response = await this.client.post<AgeSafetyBatchResult>('/v1/age-safety/check-batch', form, {
        headers: form.getHeaders(),
        maxBodyLength: Infinity,
        timeout: this.batchTimeoutMs,
      });
      return response.data;
    } catch (error) {
      this.throwUpstreamError(error, 'Batch check');
    }
  }
}
