import { Module } from '@nestjs/common';

import { AgeSafetyController } from './controllers/age-safety.controller';
import { AgeSafetyService } from './services/age-safety.service';
import { MlClientService } from './services/ml-client.service';

@Module({
  controllers: [AgeSafetyController],
  providers: [AgeSafetyService, MlClientService],
})
export class AppModule {}
