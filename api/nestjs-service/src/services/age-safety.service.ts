import { Injectable } from '@nestjs/common';

import { CheckRequestDto } from '../dto/check-request.dto';
import { AgeSafetyBatchResult, AgeSafetyHealth, AgeSafetyResult } from '../types/age-safety.types';
import { MlClientService } from './ml-client.service';

@Injectable()
export class AgeSafetyService {
  constructor(private readonly mlClientService: MlClientService) {}

  async health(): Promise<AgeSafetyHealth> {
    return this.mlClientService.health();
  }

  async checkImage(file: Express.Multer.File, metadata: CheckRequestDto): Promise<AgeSafetyResult> {
    return this.mlClientService.checkImage(file, metadata);
  }

  async checkBatch(files: Express.Multer.File[], metadata: CheckRequestDto): Promise<AgeSafetyBatchResult> {
    return this.mlClientService.checkBatch(files, metadata);
  }
}
