<template>
    <v-card outlined class="rounded-lg pa-4">
        <v-row>
            <v-col
                v-for="filePath in filePaths_"
                :key="filePath"
                cols="12"
                sm="6"
                md="4"
                lg="2"
                class="d-flex child-flex justify-center"
            >
                <v-img
                    :src="filePath"
                    aspect-ratio="1"
                    max-width="256"
                    contain
                    class="cursor-pointer"
                    @click="imgClick(filePath)"
                >
                    <template #placeholder>
                        <v-row
                            class="fill-height ma-0"
                            align="center"
                            justify="center"
                        >
                            <v-progress-circular indeterminate color="blue lighten-3" />
                        </v-row>
                    </template>
                </v-img>
            </v-col>
        </v-row>
    </v-card>
</template>

<script>
import axios from 'axios'
const apiBaseUrl = process.env.apiBaseUrl

export default {
    name: 'ListResults',
    props: {
        filePaths: {
            type: Array,
            default: () => []
        }
    },
    data() {
        return {
            filePaths_: []
        }
    },
    watch: {
        // If prop changes, `filePaths_` equals the prop:
        filePaths() {
            this.filePaths_ = this.filePaths
        }
    },
    methods: {
        imgClick(imgPath) {
            this.$emit('loading', true)
            this.$emit('error', '')

            axios.put(
                `${apiBaseUrl}/api/find-related`,
                { imgPath })
                .then((response) => {
                    this.$emit('select-img', imgPath)
                    this.filePaths_ = response?.data?.file_paths
                })
                .catch((error) => {
                    this.$emit('error', error)
                })
                .finally(() => {
                    this.$emit('loading', false)
                })
        }
    }
}
</script>
