#include "drive_guard.h"
#include <string.h>
static uint16_t get16(const uint8_t* p) { return (uint16_t)(p[0]|((uint16_t)p[1]<<8)); }
static uint32_t get32(const uint8_t* p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static void put16(uint8_t* p,uint16_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void put32(uint8_t* p,uint32_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}
uint16_t rp_crc16(const uint8_t* data,size_t length) {
    uint16_t crc=0xffff; size_t i;
    for (i=0;i<length;++i) {
        int bit; crc^=(uint16_t)((uint16_t)data[i]<<8);
        for (bit=0;bit<8;++bit) crc=(uint16_t)((crc&0x8000) ? (crc<<1)^0x1021 : crc<<1);
    }
    return crc;
}
void rp_encode(const rp_command* c,uint8_t p[RP_PACKET_SIZE]) {
    p[0]='R'; p[1]='P'; p[2]=1; p[3]=c->enable;
    put32(p+4,c->sequence); put16(p+8,c->speed_mm_s);
    put16(p+10,(uint16_t)c->steering_mrad); put16(p+12,c->lease_ms);
    put16(p+14,rp_crc16(p,14));
}
int rp_decode(const uint8_t p[RP_PACKET_SIZE],rp_command* c) {
    uint16_t steering;
    if (p[0]!='R' || p[1]!='P' || p[2]!=1 || p[3]>1 || get16(p+14)!=rp_crc16(p,14)) return 0;
    c->enable=p[3]; c->sequence=get32(p+4); c->speed_mm_s=get16(p+8);
    steering=get16(p+10);
    c->steering_mrad=(int16_t)(steering<=32767 ? (int32_t)steering : (int32_t)steering-65536);
    c->lease_ms=get16(p+12);
    return c->speed_mm_s<=RP_MAX_SPEED_MM_S &&
           c->steering_mrad>=-RP_MAX_STEERING_MRAD && c->steering_mrad<=RP_MAX_STEERING_MRAD &&
           c->lease_ms>0 && c->lease_ms<=RP_MAX_LEASE_MS &&
           (c->enable || (c->speed_mm_s==0 && c->steering_mrad==0));
}
static void stop(rp_guard* g) { g->enabled=0; g->speed_mm_s=0; g->steering_mrad=0; }
void rp_guard_init(rp_guard* g) { memset(g,0,sizeof(*g)); }
int rp_guard_arm(rp_guard* g,int physical_estop) {
    if (physical_estop) { g->estop_latched=1; g->armed=0; stop(g); return 0; }
    stop(g); g->estop_latched=0; g->armed=1; g->used=0;
    /* Keep sequence history. A Pi restart needs a board-local session reset. */
    return 1;
}
void rp_guard_tick(rp_guard* g,uint32_t now,int physical_estop) {
    if (physical_estop) { g->estop_latched=1; g->armed=0; stop(g); }
    if (!g->armed || g->estop_latched) stop(g);
    if (g->enabled && (uint32_t)(now-g->received_ms)>=g->lease_ms) {
        stop(g); g->armed=0;
    }
}
static void accept(rp_guard* g,const rp_command* c,uint32_t now) {
    const uint32_t delta=c->sequence-g->last_sequence;
    if (g->have_sequence && (delta==0 || delta>=0x80000000u)) return;
    g->have_sequence=1; g->last_sequence=c->sequence;
    if (!c->enable) { stop(g); return; }
    if (!g->armed || g->estop_latched) return;
    g->received_ms=now; g->lease_ms=c->lease_ms;
    g->enabled=1; g->speed_mm_s=c->speed_mm_s; g->steering_mrad=c->steering_mrad;
}
void rp_guard_feed(rp_guard* g,const uint8_t* bytes,size_t length,uint32_t now) {
    size_t i;
    rp_guard_tick(g,now,0); /* Expire old lease BEFORE accepting a late heartbeat. */
    if (g->used && (uint32_t)(now-g->last_byte_ms)>50) g->used=0;
    if (length) g->last_byte_ms=now;
    for (i=0;i<length;++i) {
        rp_command command;
        g->buffer[g->used++]=bytes[i];
        if (g->used<RP_PACKET_SIZE) continue;
        if (rp_decode(g->buffer,&command)) { accept(g,&command,now); g->used=0; }
        else { memmove(g->buffer,g->buffer+1,RP_PACKET_SIZE-1); g->used=RP_PACKET_SIZE-1; }
    }
}
